"""Export pooled probe scoring as ONNX logistic regression."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from probekit.onnx_utils import write_graph
from probekit.probes.raw_trainer import LinearProbe


def export_probe(probe: LinearProbe, path: Path) -> None:
    width = probe.weights.size
    weights = numpy_helper.from_array(probe.weights.astype(np.float32).reshape(width, 1), "weights")
    bias = numpy_helper.from_array(np.asarray([probe.intercept], dtype=np.float32), "bias")
    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["input", "weights", "bias"], ["logits"]),
            helper.make_node("Sigmoid", ["logits"], ["probabilities"]),
        ],
        "linear_probe",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["sequences", width])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["sequences", 1])],
        [weights, bias],
    )
    write_graph(path, graph)


def predict_onnx(session: ort.InferenceSession, matrix: np.ndarray) -> np.ndarray:
    return session.run(None, {"input": np.asarray(matrix, dtype=np.float32)})[0][:, 0]
