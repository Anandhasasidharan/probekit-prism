"""Export trained logistic-regression probes to ONNX.

Hand-built graph (no skl2onnx dependency): sigmoid(X @ W^T + b). Verified
against sklearn probabilities with onnxruntime when available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline


def extract_lr(pipeline_or_model) -> tuple[np.ndarray, np.ndarray]:
    """Return (W, b) from a Pipeline(StandardScaler, LogisticRegression),
    folding the scaler statistics into the weights."""
    if isinstance(pipeline_or_model, Pipeline):
        scaler = pipeline_or_model.named_steps["scaler"]
        lr = pipeline_or_model.named_steps["lr"]
        W = lr.coef_ / scaler.scale_
        b = lr.intercept_ - (lr.coef_ @ (scaler.mean_ / scaler.scale_)).ravel()
        return W.astype(np.float32), b.astype(np.float32)
    raise TypeError("Expected sklearn Pipeline")


def export_probe_onnx(probe, path: str | Path, opset: int = 17) -> str:
    import onnx
    from onnx import TensorProto, helper

    W, b = extract_lr(probe.model)  # W: (1, d) for binary LR
    d_in = W.shape[1]
    W_t = W.T.copy()  # (d, 1) so MatMul(X[n,d] @ W_t[d,1]) broadcasts
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    initializers = [
        helper.make_tensor("W", TensorProto.FLOAT, W_t.shape, W_t.reshape(-1).tolist()),
        helper.make_tensor("b", TensorProto.FLOAT, [b.shape[0]], b.tolist()),
    ]
    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["logits"]),
        helper.make_node("Add", ["logits", "b"], ["shifted"]),
        helper.make_node("Sigmoid", ["shifted"], ["probability"]),
    ]
    graph = helper.make_graph(
        nodes,
        "probekit_probe",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, d_in])],
        [
            helper.make_tensor_value_info("probability", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 1]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def verify_probe_onnx(path: str | Path, probe, sample: np.ndarray) -> float:
    """Max abs difference between sklearn and ONNX probabilities."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = probe.predict_proba(sample.astype(np.float32))
    got = sess.run(["probability"], {"X": sample.astype(np.float32)})[0].reshape(-1)
    return float(np.abs(got - expected).max())
