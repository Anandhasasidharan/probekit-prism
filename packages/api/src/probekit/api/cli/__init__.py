"""ProbeKit CLI: management commands for the full pipeline.

Usage (after `uv sync`):
    uv run probekit harvest --prompts data/examples/prompts.jsonl
    uv run probekit train-sae
    uv run probekit auto-interp --feature 42
    uv run probekit train-probe --concept pii_detection
    uv run probekit steer --prompt "delete all files in /data"
    uv run probekit monitor
    uv run probekit serve
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any


def _load_config(args):
    from probekit.core.config import GlobalConfig

    return GlobalConfig.from_yaml(getattr(args, "config", None) or "./config.yaml")


# ------------------------------------------------------------------- commands

def cmd_harvest(args) -> int:
    import asyncio

    from probekit.harvest import ActivationHarvester

    cfg = _load_config(args).harvest
    harvester = ActivationHarvester(
        model_name=cfg.model_name,
        capture_layers=cfg.capture_layers,
        batch_size=cfg.batch_size,
        device=args.device,
    )
    if args.prompts:
        paths = asyncio.run(
            harvester.harvest_batch(
                args.prompts, args.output, max_sequences=args.max
            )
        )
        print(json.dumps({"batches_written": len(paths), "output": args.output}, indent=2))
    else:
        prompts = [line.strip() for line in sys.stdin if line.strip()]
        out = args.output or "./data/activations/stdin.parquet"
        records = harvester.harvest_sync(prompts, save_path=out)
        print(f"Wrote {len(records)} activation records")
    return 0


def cmd_train_sae(args) -> int:
    from probekit.harvest.compression import load_layer_matrix
    from probekit.sae.architectures import build_sae
    from probekit.sae.evaluator import evaluate_sae
    from probekit.sae.exporter import export_sae_onnx
    from probekit.sae.trainer import SAETrainer

    cfg = _load_config(args).sae
    matrix, tokens = load_layer_matrix(args.data, layer=cfg.layer)

    sae = build_sae(cfg.architecture, d_model=matrix.shape[1],
                    dict_size=cfg.dict_size, k=cfg.k,
                    sparsity_coef=cfg.sparsity_coef, device=args.device)
    trainer = SAETrainer(sae, lr=cfg.lr, device=args.device)
    trainer.train(matrix, epochs=args.epochs)

    metrics = evaluate_sae(sae, matrix, device=args.device)
    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_frequencies"}, indent=2))

    trainer.save(args.out + ".pt")
    export_sae_onnx(sae, args.out + ".onnx")
    print(f"Saved {args.out}.pt / .onnx")
    return 0


def cmd_auto_interp(args) -> int:
    from probekit.auto_interp.pipeline import AutoInterpPipeline
    from probekit.harvest.compression import load_layer_matrix
    from probekit.sae.evaluator import encode_matrix

    cfg = _load_config(args)
    matrix, tokens = load_layer_matrix(args.data, layer=cfg.sae.layer)

    import torch

    sae_state = torch.load(args.sae + ".pt", map_location="cpu", weights_only=True)
    from probekit.sae.architectures import build_sae

    sae = build_sae("topk", d_model=sae_state["W_enc"].shape[0],
                    dict_size=sae_state["W_enc"].shape[1], k=cfg.sae.k)
    sae.load_state_dict(sae_state)

    latents = encode_matrix(sae, matrix)
    pipeline = AutoInterpPipeline(provider=cfg.auto_interp.provider, model=cfg.auto_interp.model,
                                  cache_db=cfg.auto_interp.cache_db)
    n_feats = min(args.top_features, latents.shape[1])
    feature_ids = [args.feature] if args.feature else list(range(n_feats))
    for fid in feature_ids:
        result = pipeline.run_feature(fid, latents, tokens)
        print(f"[{fid}] {result['text']}")
    pipeline.close()
    return 0


def cmd_train_probe(args) -> int:
    """Train all three probe kinds on a concept JSONL and report comparison."""
    from pathlib import Path

    from probekit.harvest import ActivationHarvester
    from probekit.probes.comparator import ThreeWayComparator
    from probekit.probes.dataset import build_dataset_from_records, load_examples

    cfg = _load_config(args)
    examples = load_examples(args.concept_file)
    concept = args.concept or examples[0]["concept"]

    # Harvest activations for the example texts.
    harvester = ActivationHarvester(model_name=cfg.harvest.model_name,
                                    capture_layers=[cfg.sae.layer],
                                    device=args.device)
    records = harvester.harvest_sync([ex["text"] for ex in examples])
    ds = build_dataset_from_records(records, examples, concept)
    train, test, ood = ds.split(test_size=cfg.probes.test_split, ood_size=cfg.probes.ood_split)

    comparator = ThreeWayComparator(device=args.device)  # SAE/CLT optional
    comparator.compare_concept(train, test)
    print(comparator.table())
    comparator.report(Path(f"./data/probes/{concept}_comparison.json"))
    return 0


def cmd_steer(args) -> int:
    from probekit.steering.controller import SteeringController
    from probekit.steering.interventions import SteeringIntervention

    cfg = _load_config(args)
    strategy_cfg = cfg.steering.strategies[args.strategy]

    from probekit.core.device import get_device
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device(args.device)
    model: Any = AutoModelForCausalLM.from_pretrained(cfg.harvest.model_name)
    model.to(device=str(device))
    model.eval()
    tokenizer: Any = AutoTokenizer.from_pretrained(cfg.harvest.model_name)

    fid = args.feature_id if args.feature_id is not None else 0
    direction = torch_decoder_direction(cfg, fid, device)
    iv = SteeringIntervention(direction, operation=strategy_cfg.intervention,
                              strength=strategy_cfg.strength)
    controller = SteeringController(model, tokenizer, intervention=iv,
                                    layers=[min(5, model.config.n_layer - 1)])
    out = controller.generate(args.prompt, max_new_tokens=args.max_tokens, steer=not args.no_steer)
    print(out)
    return 0


def torch_decoder_direction(cfg, feature_id: int, device):
    import torch

    state = torch.load("./data/saes/sae.pt", map_location=device, weights_only=True)
    return state["W_dec"][feature_id].detach()


def cmd_monitor(args) -> int:
    from fastapi.testclient import TestClient  # noqa: F401 - optional path
    from probekit.api.main import create_app

    app = create_app(args.config)
    with TestClient(app) as client:
        resp = client.get("/monitor/drift")
        print(json.dumps(resp.json(), indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    cfg = _load_config(args).api
    uvicorn.run("probekit.api.main:create_app", host=cfg.host, port=cfg.port,
                factory=True, log_level=cfg.log_level)
    return 0


# ---------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probekit", description="ProbeKit v2 CLI")
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("harvest", help="harvest activations")
    p.add_argument("--prompts", help="JSONL file of prompts (or stdin)")
    p.add_argument("--output", default="./data/activations/harvest.parquet")
    p.add_argument("--max", type=int, default=1000)
    p.set_defaults(func=cmd_harvest)

    p = sub.add_parser("train-sae", help="train an SAE on harvested activations")
    p.add_argument("--data", required=True, help="Parquet of activations")
    p.add_argument("--out", default="./data/saes/sae")
    p.add_argument("--epochs", type=int, default=5)
    p.set_defaults(func=cmd_train_sae)

    p = sub.add_parser("auto-interp", help="auto-label features")
    p.add_argument("--data", required=True)
    p.add_argument("--sae", default="./data/saes/sae")
    p.add_argument("--feature", type=int, default=None)
    p.add_argument("--top-features", type=int, default=20)
    p.set_defaults(func=cmd_auto_interp)

    p = sub.add_parser("train-probe", help="train raw/SAE/circuit probes and compare")
    p.add_argument("--concept-file", required=True, help="JSONL with text/label/concept")
    p.add_argument("--concept", default=None)
    p.set_defaults(func=cmd_train_probe)

    p = sub.add_parser("steer", help="apply a steering strategy to a prompt")
    p.add_argument("--prompt", required=True)
    p.add_argument("--strategy", default="suppress_risky")
    p.add_argument("--feature-id", dest="feature_id", type=int, default=None)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=32)
    p.add_argument("--no-steer", action="store_true")
    p.set_defaults(func=cmd_steer)

    p = sub.add_parser("monitor", help="run a drift check via the API app")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("serve", help="run the API server")
    p.set_defaults(func=cmd_serve)

    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
