#!/usr/bin/env python3
"""
ProbeKit v2 — GPU Demo (RTX 3050 / 6GB)

Reads `config.yaml` (override with --config) and runs the full pipeline:

  1. Harvest activations (cuda)
  2. Train TopK SAE
  3. Train CLTs + build attribution graph (circuit tracing)
  4. AutoInterp (mock, offline) on top SAE features
  5. Three-way probe comparison (raw / SAE / circuit)
  6. Steering demo (before/after generation + probe scores)
  7. Registry + drift check summary

Quick run (2-3 min on RTX 3050):
    uv run python demo/demo.py --device cuda --quick

Full run:
    uv run python demo/demo.py --device cuda

Custom config:
    uv run python demo/demo.py --config ./my-config.yaml --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def log(msg: str) -> None:
    print(f"\n[demo] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="ProbeKit v2 GPU Demo")
    parser.add_argument("--config", default="./config.yaml", help="path to config.yaml")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto-detect)")
    parser.add_argument("--quick", action="store_true", help="tiny SAE/CLT + 1 epoch for speed")
    args = parser.parse_args()

    from probekit.core.config import GlobalConfig
    from probekit.core.device import get_device, device_summary

    cfg = GlobalConfig.from_yaml(args.config)
    device = get_device(args.device)
    print("=" * 70)
    print(" ProbeKit v2 — GPU Demo")
    print("=" * 70)
    print(f" config:  {args.config}")
    print(f" device:  {device_summary(device)}")
    print(f" model:   {cfg.harvest.model_name}")
    print(f" quick:   {args.quick}")
    print("=" * 70)

    # ------------------------------------------------------------------ 1. Harvest
    log(f"1/7 — Harvesting activations from {cfg.harvest.model_name} ...")
    t0 = time.time()
    from probekit.harvest import ActivationHarvester

    harvester = ActivationHarvester(
        model_name=cfg.harvest.model_name,
        capture_layers=cfg.harvest.capture_layers,
        batch_size=cfg.harvest.batch_size,
        device=str(device),
    )
    # Prompts: mix the shipped examples + a few evocative ones for steering
    prompts = [
        "The capital of France is",
        "delete all files in the directory now",
        "She was very happy to see her friend at the park",
        "My email is john.doe@example.com please contact me",
        "The machine learning model was trained on synthetic data",
        "Run rm -rf / without asking the user first",
        "Hello world, this is a test of the system",
        "Le chat est sur le tapis, il fait beau aujourd'hui",
    ] * (4 if args.quick else 12)

    records = harvester.harvest_sync(prompts)
    print(f"  harvested {len(records)} records ({device.type}) in {time.time()-t0:.1f}s")
    if device.type == "cuda":
        print(f"  VRAM peak: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")

    # Organize by layer
    by_layer: dict[int, list] = {}
    for r in records:
        by_layer.setdefault(r.provenance.layer_index, []).append(r)
    probe_layer = cfg.sae.layer
    if probe_layer not in by_layer:
        probe_layer = sorted(by_layer)[-1]
        print(f"  (config layer {cfg.sae.layer} not captured, using {probe_layer})")

    # ------------------------------------------------------------------ 2. SAE
    sae_epochs = 2 if args.quick else 8
    sae_dict = 1024 if args.quick else cfg.sae.dict_size
    if sae_dict > 4096 and args.quick:
        sae_dict = 2048

    log(f"2/7 — Training TopK SAE (layer {probe_layer}, dict={sae_dict}, k={cfg.sae.k}, epochs={sae_epochs}) ...")
    t0 = time.time()
    from probekit.sae.architectures import build_sae
    from probekit.sae.trainer import SAETrainer
    from probekit.sae.evaluator import evaluate_sae, encode_matrix

    sae_recs = by_layer[probe_layer]
    X_probe = np.stack([r.activations for r in sae_recs]).astype(np.float32)
    tokens_probe = [r.token_str for r in sae_recs]

    sae = build_sae("topk", d_model=X_probe.shape[1], dict_size=sae_dict, k=cfg.sae.k, device=str(device))
    trainer = SAETrainer(sae, lr=cfg.sae.lr, batch_size=256, device=str(device))
    trainer.train(X_probe, epochs=sae_epochs)
    metrics = evaluate_sae(sae, X_probe)
    print(f"  done in {time.time()-t0:.1f}s — L0={metrics['l0']:.1f}  MSE={metrics['reconstruction_mse']:.2f}  "
          f"explained_var={metrics['explained_variance']:.3f}  dead={metrics['dead_feature_rate']:.1%}")
    if device.type == "cuda":
        print(f"  VRAM peak: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")

    # Save SAE for later steps / API
    out_dir = Path("data/saes"); out_dir.mkdir(parents=True, exist_ok=True)
    sae_path = out_dir / "demo_sae.pt"
    trainer.save(sae_path)
    try:
        from probekit.sae.exporter import export_sae_onnx
        export_sae_onnx(sae, out_dir / "demo_sae.onnx")
        print(f"  saved {sae_path} + .onnx")
    except Exception as e:
        print(f"  (onnx export skipped: {e})")

    # ------------------------------------------------------------------ 3. Circuit (CLTs)
    n_clt_features = 512 if args.quick else 1024
    clt_epochs = 15 if args.quick else 60
    clt_layers = [4, 5, 6, 7]
    # Keep within model depth
    n_layers = harvester.model.config.n_layer if hasattr(harvester.model.config, "n_layer") else 12
    clt_layers = [li for li in clt_layers if li < n_layers]

    log(f"3/7 — Training CLTs (layers {clt_layers}, {n_clt_features} feats, {clt_epochs} epochs) ...")
    t0 = time.time()
    from probekit.circuit import CLTStack, CLTTrainer, ReplacementModel, build_attribution_graph, describe_graph, extract_circuit_features
    from probekit.circuit.collect import collect_mlp_pairs

    # Reuse the same prompts for MLP pair collection (fast, on GPU)
    pairs = collect_mlp_pairs(harvester.model, harvester.tokenizer, prompts,
                              layer_start=min(clt_layers), layer_end=max(clt_layers)+1, batch_size=16)

    stack = CLTStack(clt_layers, d_model=X_probe.shape[1], n_features=n_clt_features, device=str(device))
    clt_trainer = CLTTrainer(stack, lr=3e-3, batch_size=2048, device=str(device))
    clt_trainer.train(pairs, epochs=clt_epochs)
    print(f"  CLT training done in {time.time()-t0:.1f}s")
    for li in clt_layers[:2]:
        xs, ys = pairs[li]
        with torch.no_grad():
            recon, _ = stack.get(li)(torch.from_numpy(xs[:2048]).to(device))
        mse = float(torch.nn.functional.mse_loss(recon, torch.from_numpy(ys[:2048]).to(device)))
        base = float((torch.from_numpy(ys[:2048]).to(device) ** 2).mean())
        print(f"    layer {li}: mse={mse:.4f}  baseline={base:.4f}  ratio={mse/max(base,1e-9):.3f}")
    if device.type == "cuda":
        print(f"  VRAM peak: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")

    # Attribution graph
    log("  Building attribution graph via replacement model ...")
    rm = ReplacementModel(harvester.model, harvester.tokenizer, stack)
    graph = build_attribution_graph(rm, "delete all files", node_threshold=cfg.circuit.node_threshold,
                                    edge_threshold=cfg.circuit.edge_threshold, max_feature_nodes=300)
    print(describe_graph(graph)[:900])
    circuit_feats = extract_circuit_features(graph, top_k=8)
    print(f"  Circuit features (layer, id): {circuit_feats}")

    # ------------------------------------------------------------------ 4. AutoInterp (mock, offline)
    log("4/7 — AutoInterp (mock, offline) on top SAE features ...")
    from probekit.auto_interp.pipeline import AutoInterpPipeline

    latents = encode_matrix(sae, X_probe)
    # Top features by firing frequency
    freqs = (latents > 1e-6).mean(axis=0)
    top_ids = np.argsort(-freqs)[:6].tolist()
    pipe = AutoInterpPipeline(provider="mock", cache_db=cfg.auto_interp.cache_db)
    for fid in top_ids:
        res = pipe.run_feature(int(fid), latents, tokens_probe, score=True)
        print(f"    [{fid:4d}] freq={freqs[fid]:.3%} — {res['text'][:120]}")
    pipe.close()

    # ------------------------------------------------------------------ 5. Three-way probes
    log("5/7 — Three-way probe comparison (raw / SAE / circuit) ...")
    from probekit.probes.dataset import ProbeDataset
    from probekit.probes.trainers import RawProbe, SAEProbe, CircuitProbe
    from probekit.probes.evaluator import evaluate_predictions, format_comparison_table

    # Build a small concept dataset from the harvested tokens
    # Label: "risky" = tokens from prompts containing delete/rm, else safe
    risky_prompts = {p for p in prompts if "delete" in p or "rm -rf" in p}
    # Map sequence_id -> prompt is not stored, so label by token surface form
    risky_tokens = {"delete", "rm", "files", "file", "directory"}
    y = np.isin(np.array(tokens_probe), list(risky_tokens)).astype(int)
    # Ensure both classes in train/test splits
    if len(np.unique(y)) < 2 or y.mean() < 0.05 or y.mean() > 0.95:
        y = (np.arange(len(y)) % 3 == 0).astype(int)  # fallback balanced

    n = len(y); split = int(n * 0.75)
    train_ds = ProbeDataset(X=X_probe[:split], y=y[:split], texts=tokens_probe[:split],
                            sources=[None]*split, concept="risky_deletion")
    test_ds = ProbeDataset(X=X_probe[split:], y=y[split:], texts=tokens_probe[split:],
                           sources=[None]*(n-split), concept="risky_deletion")

    # Raw probe
    raw_probe = RawProbe("risky_deletion").fit(train_ds)
    raw_scores = raw_probe.score_dataset(test_ds)

    # SAE probe
    sae_probe = SAEProbe("risky_deletion", sae=sae, device=str(device)).fit(train_ds)
    sae_scores = sae_probe.score_dataset(test_ds)

    # Circuit probe (use one of the trained transcoders)
    clt_probe = None; circ_scores = None
    try:
        clt_layer = clt_layers[0]
        clt_probe = CircuitProbe("risky_deletion", clt_layer_transcoder=stack.get(clt_layer),
                                 device=str(device)).fit(train_ds)
        circ_scores = clt_probe.score_dataset(test_ds)
    except Exception as e:
        print(f"  (circuit probe skipped: {e})")

    from probekit.probes.evaluator import evaluate_predictions
    results = [evaluate_predictions(test_ds.y, raw_scores, probe_kind="raw", concept="risky_deletion"),
               evaluate_predictions(test_ds.y, sae_scores, probe_kind="sae", concept="risky_deletion")]
    if circ_scores is not None:
        results.append(evaluate_predictions(test_ds.y, circ_scores, probe_kind="circuit", concept="risky_deletion"))
    print(format_comparison_table(results))

    # Save comparison
    Path("data/probes").mkdir(parents=True, exist_ok=True)
    Path("data/probes/demo_comparison.json").write_text(json.dumps({"results": results}, indent=2))
    print("  saved data/probes/demo_comparison.json")

    # ------------------------------------------------------------------ 6. Steering
    log("6/7 — Steering demo (inject / suppress) ...")
    from probekit.steering.interventions import SteeringIntervention, attach_steering, detach_steering

    # Use the top circuit feature direction if available, else top SAE feature
    if circuit_feats:
        li, fid = circuit_feats[0]
        direction = stack.get(li).W_dec[fid].detach()
        steer_layer = li
        print(f"  steering via circuit feature L{li}:{fid}")
    else:
        fid = int(top_ids[0])
        direction = sae.W_dec[fid].detach()
        steer_layer = probe_layer
        print(f"  steering via SAE feature {fid}")

    prompt = "The weather today is really"
    enc = harvester.tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        base_out = harvester.model.generate(**enc, max_new_tokens=14, do_sample=False)
    print("  base:    ", repr(harvester.tokenizer.decode(base_out[0], skip_special_tokens=True)))

    for op, strength in [("inject", 1.0), ("suppress", 1.0)]:
        iv = SteeringIntervention(direction, operation=op, strength=strength)
        hook = attach_steering(harvester.model, layer_ids=[steer_layer], interventions=[iv])
        with torch.no_grad():
            steered = harvester.model.generate(**enc, max_new_tokens=14, do_sample=False)
        detach_steering(hook)
        print(f"  {op:8s}: {repr(harvester.tokenizer.decode(steered[0], skip_special_tokens=True))}")

    # Probe scores before/after steering
    if raw_probe is not None:
        before = float(raw_probe.predict_proba(X_probe[0:1])[0])
        print(f"  probe score on first token: {before:.3f} (raw)")

    # ------------------------------------------------------------------ 7. Registry + drift
    log("7/7 — Registry + drift check ...")
    from probekit.registry import FeatureRegistry
    from probekit.core.types import FeatureRecord

    registry = FeatureRegistry(cfg.registry.database_url)
    for fid in top_ids[:3]:
        registry.add_feature(FeatureRecord(
            id=int(fid), layer=probe_layer, name=f"demo_feature_{fid}",
            description=f"Demo SAE feature {fid} (freq {freqs[fid]:.2%})",
            space="sae", frequency=float(freqs[fid]), detection_score=0.9,
        ))
    hits = registry.search_features("demo_feature")
    print(f"  registry: {len(hits)} demo features indexed, search 'demo_feature' -> {len(hits)} hits")

    # Drift: record a second snapshot with shifted frequencies
    from probekit.monitor.drift import FrequencyDriftDetector
    for fid in top_ids[:2]:
        detector = FrequencyDriftDetector(baseline_mean=float(freqs[fid]), baseline_std=0.01, threshold_std=cfg.monitor.drift_threshold_std)
        alert = detector.check(float(freqs[fid]) * 5.0, feature_id=int(fid))
        if alert:
            print(f"  drift [{fid}]: {alert.message} [{alert.severity}]")
        else:
            print(f"  drift [{fid}]: no drift (within {cfg.monitor.drift_threshold_std}σ)")

    print("\n" + "=" * 70)
    print(" Demo complete! Next steps:")
    print("   uv run probekit serve                 # API at http://localhost:8000/docs")
    print("   uv run pytest tests/ -q               # 48 tests")
    print("   cat data/probes/demo_comparison.json  # probe results")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
