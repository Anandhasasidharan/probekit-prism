"""Attribution graph construction via backward Jacobian tracing.

Node attributions: with the replacement model, the target logit is an
(affine) function of feature activations for a fixed input, so a single
backward pass yields exact local gradients  ∂logit_target / ∂a_f.  The node's
attribution score is grad * activation (its actual contribution).

Edge weights: estimated with "virtual weights" (Anthropic's published
approximation) — the direct linear path from upstream decoder direction u to
downstream encoder direction v:  W_dec[u] · W_enc[v], weighted by the upstream
activation. Edges are pruned by percentile thresholds from CircuitConfig.
"""

from __future__ import annotations

import torch
from probekit.circuit.replacement import ReplacementModel
from probekit.core.types import AttributionEdge, AttributionGraph, AttributionNode


@torch.enable_grad()
def build_attribution_graph(
    replacement: ReplacementModel,
    prompt: str,
    *,
    target_token: str | None = None,
    max_feature_nodes: int = 7500,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.95,
    top_edges_per_node: int = 8,
) -> AttributionGraph:
    """Build a pruned attribution graph for `prompt` -> target token logit.

    node_threshold: keep nodes whose |attribution| >= this quantile of all
    candidate nodes. edge_threshold: keep edges above this quantile of edge
    weight magnitude within each destination layer pair.
    """
    enc = replacement.tokenizer(prompt, return_tensors="pt").to(replacement.device)
    run = replacement.run(prompt=prompt, enc=enc)

    pos = run.last_position
    logits = run.logits[0, pos]

    if target_token is None:
        target_idx = int(logits.argmax().item())
        target_token = replacement.tokenizer.decode([target_idx])
    else:
        ids = replacement.tokenizer.encode(target_token, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Cannot tokenize target token {target_token!r}")
        target_idx = ids[0]

    scalar = logits[target_idx]
    grads = torch.autograd.grad(scalar, list(run.feature_acts.values()))
    grad_by_layer = dict(zip(run.feature_acts.keys(), grads))

    # ------------------------------------------------------------- collect nodes
    candidates: list[tuple[int, int, float, float]] = []  # (layer, feat, act, attr)
    for li in sorted(run.feature_acts):
        a = run.feature_acts[li][0, pos].detach()
        g = grad_by_layer[li][0, pos].detach()
        attr = (a * g).float()
        mask = a > 1e-6
        for f in mask.nonzero(as_tuple=True)[0].tolist():
            candidates.append((li, f, float(a[f]), float(attr[f])))

    if not candidates:
        return AttributionGraph(prompt=prompt, input_token=target_token, target_logit=target_idx)

    # Prune to top-N by |attribution|, then quantile threshold.
    candidates.sort(key=lambda t: -abs(t[3]))
    candidates = candidates[:max_feature_nodes]
    attrs = torch.tensor([abs(c[3]) for c in candidates])
    cutoff = (
        torch.quantile(attrs, 1.0 - node_threshold).item()
        if node_threshold < 1.0
        else 0.0
    )
    kept = [c for c in candidates if abs(c[3]) >= cutoff]

    graph = AttributionGraph(
        prompt=prompt,
        input_token=target_token,
        target_logit=target_idx,
        nodes=[
            AttributionNode(feature_id=f, layer=li, activation=a, attribution=t)
            for li, f, a, t in kept
        ],
    )

    # ------------------------------------------------------------------ edges
    by_layer: dict[int, list[int]] = {}
    node_lookup = {(n.layer, n.feature_id): n for n in graph.nodes}
    for n in graph.nodes:
        by_layer.setdefault(n.layer, []).append(n.feature_id)
    layers_sorted = sorted(by_layer)

    stack = replacement.stack
    edges: list[AttributionEdge] = []
    for i, l_src in enumerate(layers_sorted):
        src_feats = by_layer[l_src]
        w_dec = stack.get(l_src).W_dec.detach()  # (F_src, d)
        u_vecs = torch.stack([w_dec[f] for f in src_feats])  # (S, d)

        for l_dst in layers_sorted[i + 1:]:
            dst_feats = by_layer[l_dst]
            w_enc = stack.get(l_dst).W_enc.detach()  # (d, F_dst)
            v_vecs = torch.stack([w_enc[:, f] for f in dst_feats])  # (D, d)

            virt = u_vecs @ v_vecs.T  # (S, D) virtual weight matrix
            src_acts = torch.tensor(
                [node_lookup[(l_src, f)].activation for f in src_feats],
                device=virt.device,
            )
            weights = virt.clamp(min=0) * src_acts.unsqueeze(1)

            k = min(top_edges_per_node, weights.shape[1])
            vals, cols = torch.topk(weights, k, dim=1)
            row_cut = (
                torch.quantile(vals.flatten(), 1.0 - edge_threshold)
                if edge_threshold < 1
                else 0.0
            )

            for s in range(weights.shape[0]):
                for d_i in range(k):
                    w = float(vals[s, d_i])
                    if w <= 0 or w < row_cut:
                        continue
                    edges.append(
                        AttributionEdge(
                            src_feature=src_feats[s],
                            src_layer=l_src,
                            dst_feature=dst_feats[int(cols[s, d_i])],
                            dst_layer=l_dst,
                            weight=w,
                        )
                    )
    graph.edges = edges
    return graph
