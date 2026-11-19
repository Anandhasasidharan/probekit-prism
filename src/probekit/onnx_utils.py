"""Small ONNX graphs built with explicit dynamic batch dimensions."""

from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import onnxruntime as ort
from onnx import GraphProto, helper


def write_graph(path: Path, graph: GraphProto) -> None:
    model = helper.make_model(
        graph,
        producer_name="probekit",
        ir_version=9,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        onnx.save_model(model, temporary)
        temporary.replace(path)


def cpu_session(path: Path, threads: int = 1) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
