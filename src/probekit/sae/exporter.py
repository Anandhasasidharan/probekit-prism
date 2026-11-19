"""Export only the SAE encoder affine operations, with NumPy TopK at runtime."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from probekit.onnx_utils import write_graph
from probekit.sae.architectures import TopKSAE


def export_encoder(sae: TopKSAE, path: Path) -> None:
    parameters = {
        "scale": sae.scale.detach().cpu().numpy(),
        "center": sae.decoder.bias.detach().cpu().numpy(),
        "weights": sae.encoder.weight.detach().cpu().numpy().T,
        "bias": sae.encoder.bias.detach().cpu().numpy(),
    }
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["input", "scale"], ["scaled"]),
            helper.make_node("Sub", ["scaled", "center"], ["centered"]),
            helper.make_node("MatMul", ["centered", "weights"], ["products"]),
            helper.make_node("Add", ["products", "bias"], ["preactivations"]),
        ],
        "sae_encoder",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["tokens", sae.d_model])],
        [
            helper.make_tensor_value_info(
                "preactivations",
                TensorProto.FLOAT,
                ["tokens", sae.dict_size],
            )
        ],
        [
            numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)
            for name, value in parameters.items()
        ],
    )
    write_graph(path, graph)


def topk_numpy(preactivations: np.ndarray, k: int) -> np.ndarray:
    if not 0 < k <= preactivations.shape[-1]:
        raise ValueError("invalid TopK sparsity")
    indices = np.argpartition(-preactivations, k - 1, axis=-1)[..., :k]
    values = np.take_along_axis(preactivations, indices, axis=-1)
    values = np.maximum(values, np.finfo(preactivations.dtype).tiny)
    output = np.zeros_like(preactivations)
    np.put_along_axis(output, indices, values, axis=-1)
    return output


def encode_onnx(session: ort.InferenceSession, matrix: np.ndarray, k: int) -> np.ndarray:
    pre = session.run(None, {"input": np.asarray(matrix, dtype=np.float32)})[0]
    return topk_numpy(pre, k)
