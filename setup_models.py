"""
Download / create ONNX models for NPU-accelerated content analysis.

 * Image model  — MobileNetV2 (pre-trained, ~14 MB download)
 * Text model   — Small MLP classifier (created locally)

Run once:  python setup_models.py
"""

import importlib.util
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Load ONNX protobuf defs directly (bypasses broken C ext on Py 3.14) ────
_site = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
_pb2_path = _site / "onnx" / "onnx_ml_pb2.py"
if not _pb2_path.exists():
    # Try alternate location (Unix-style)
    import sysconfig
    _site2 = Path(sysconfig.get_path("purelib"))
    _pb2_path = _site2 / "onnx" / "onnx_ml_pb2.py"

_spec = importlib.util.spec_from_file_location("onnx_pb2", str(_pb2_path))
onnx_pb2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(onnx_pb2)

TensorProto = onnx_pb2.TensorProto


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers — build ONNX model from raw protobufs
# ═══════════════════════════════════════════════════════════════════════════

def _make_tensor(name: str, arr: np.ndarray) -> "onnx_pb2.TensorProto":
    """Create an initializer TensorProto from a numpy array."""
    tp = onnx_pb2.TensorProto()
    tp.name = name
    tp.data_type = TensorProto.FLOAT
    tp.dims.extend(arr.shape)
    tp.raw_data = arr.astype(np.float32).tobytes()
    return tp


def _make_node(op: str, inputs: list[str], outputs: list[str], name: str = "") -> "onnx_pb2.NodeProto":
    """Create a NodeProto."""
    np_ = onnx_pb2.NodeProto()
    np_.op_type = op
    np_.input.extend(inputs)
    np_.output.extend(outputs)
    if name:
        np_.name = name
    return np_


def _make_value_info(name: str, elem_type: int, shape: list) -> "onnx_pb2.ValueInfoProto":
    """Create a ValueInfoProto for graph inputs/outputs."""
    vi = onnx_pb2.ValueInfoProto()
    vi.name = name
    vi.type.tensor_type.elem_type = elem_type  # 1 = FLOAT
    for dim in shape:
        d = vi.type.tensor_type.shape.dim.add()
        if isinstance(dim, str):
            d.dim_param = dim
        else:
            d.dim_value = dim
    return vi


def _save_model(graph, path: str, opset: int = 15):
    """Assemble a ModelProto and serialise to disk."""
    model = onnx_pb2.ModelProto()
    model.ir_version = 8
    oi = model.opset_import.add()
    oi.domain = ""
    oi.version = opset
    model.graph.CopyFrom(graph)
    with open(path, "wb") as f:
        f.write(model.SerializeToString())
    size = os.path.getsize(path)
    print(f"  Saved {path} ({size:,} bytes)")


# ═══════════════════════════════════════════════════════════════════════════
#  1. Image Model — MobileNetV2 Download
# ═══════════════════════════════════════════════════════════════════════════

MOBILENET_URLS = [
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
    "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
]


def download_image_model() -> bool:
    dest = MODELS_DIR / "mobilenetv2-12.onnx"
    if dest.exists() and dest.stat().st_size > 5_000_000:
        print(f"  MobileNetV2 already present ({dest.stat().st_size:,} bytes)")
        return True

    for url in MOBILENET_URLS:
        try:
            print(f"  Downloading MobileNetV2 from:\n    {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) < 5_000_000:
                print(f"  File too small ({len(data):,} bytes) — trying next URL")
                continue
            dest.write_bytes(data)
            print(f"  Downloaded: {len(data):,} bytes")
            return True
        except Exception as e:
            print(f"  Failed: {e}")

    print("  ⚠ Could not download MobileNetV2. Will create custom image CNN.")
    return False


def create_custom_image_model():
    """
    Create a small custom image feature network.
    Input: (1, 3, 224, 224) — standard ImageNet-sized RGB image
    Output: (1, 1000) — pseudo-class logits

    Architecture (all FC, no conv for protobuf simplicity):
      Flatten → FC(150528, 512) → ReLU → FC(512, 256) → ReLU → FC(256, 1000)

    NOTE: For a demo this is a real neural network doing real matrix multiplications
    on the NPU.  With random He-initialised weights the activations are well-scaled
    but the predictions are not semantically meaningful — that's fine because we use
    the output distribution statistics (entropy, concentration) as a signal, not the
    class labels.
    """
    dest = MODELS_DIR / "image_model.onnx"
    print("  Creating custom image feature network …")

    np.random.seed(42)

    # We'll work on 64×64 resized images to keep weights manageable
    # Input: (1, 3, 64, 64)  → flatten → 12288
    # FC1: 12288 → 256, FC2: 256 → 128, FC3: 128 → 64 (features)
    INPUT_DIM = 3 * 64 * 64  # 12288

    W1 = (np.random.randn(INPUT_DIM, 256) * np.sqrt(2.0 / INPUT_DIM)).astype(np.float32)
    B1 = np.zeros(256, dtype=np.float32)
    W2 = (np.random.randn(256, 128) * np.sqrt(2.0 / 256)).astype(np.float32)
    B2 = np.zeros(128, dtype=np.float32)
    W3 = (np.random.randn(128, 64) * np.sqrt(2.0 / 128)).astype(np.float32)
    B3 = np.zeros(64, dtype=np.float32)

    graph = onnx_pb2.GraphProto()
    graph.name = "image_feature_net"

    # Initializers (weights)
    graph.initializer.append(_make_tensor("W1", W1))
    graph.initializer.append(_make_tensor("B1", B1))
    graph.initializer.append(_make_tensor("W2", W2))
    graph.initializer.append(_make_tensor("B2", B2))
    graph.initializer.append(_make_tensor("W3", W3))
    graph.initializer.append(_make_tensor("B3", B3))

    # Shape constant for Reshape (flatten)
    shape_val = np.array([1, INPUT_DIM], dtype=np.int64)
    shape_tp = onnx_pb2.TensorProto()
    shape_tp.name = "flat_shape"
    shape_tp.data_type = TensorProto.INT64
    shape_tp.dims.append(2)
    shape_tp.raw_data = shape_val.tobytes()
    graph.initializer.append(shape_tp)

    # Nodes
    graph.node.append(_make_node("Reshape", ["input", "flat_shape"], ["flat"], "reshape"))
    graph.node.append(_make_node("MatMul", ["flat", "W1"], ["mm1"], "matmul1"))
    graph.node.append(_make_node("Add", ["mm1", "B1"], ["fc1"], "add1"))
    graph.node.append(_make_node("Relu", ["fc1"], ["relu1"], "relu1"))
    graph.node.append(_make_node("MatMul", ["relu1", "W2"], ["mm2"], "matmul2"))
    graph.node.append(_make_node("Add", ["mm2", "B2"], ["fc2"], "add2"))
    graph.node.append(_make_node("Relu", ["fc2"], ["relu2"], "relu2"))
    graph.node.append(_make_node("MatMul", ["relu2", "W3"], ["mm3"], "matmul3"))
    graph.node.append(_make_node("Add", ["mm3", "B3"], ["output"], "add3"))

    # IO
    graph.input.append(_make_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64]))
    graph.output.append(_make_value_info("output", TensorProto.FLOAT, [1, 64]))

    _save_model(graph, str(dest))
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  2. Text Classifier Model — Small MLP
# ═══════════════════════════════════════════════════════════════════════════

def create_text_model():
    """
    Create a small MLP for text feature classification.
    Input:  (1, 6)  — six normalised heuristic feature scores
    Output: (1, 1)  — AI probability (after sigmoid)

    Architecture: FC(6→64) → ReLU → FC(64→32) → ReLU → FC(32→1) → Sigmoid

    Weights are initialised to approximate our current weighted-average scoring
    but with genuine non-linear transformations running on the NPU.
    """
    dest = MODELS_DIR / "text_classifier.onnx"
    print("  Creating text classifier MLP …")

    np.random.seed(2024)

    # Layer 1: 6 → 64
    W1 = (np.random.randn(6, 64) * np.sqrt(2.0 / 6)).astype(np.float32)
    # Seed key neurons with our tuned weight patterns
    W1[:, 0] = [0.40, 0.40, 0.30, 0.20, 0.40, 0.30]   # Weighted-average neuron
    W1[:, 1] = [0.50, -0.30, 0.40, 0.10, 0.30, 0.20]   # Vocab+repetition detector
    W1[:, 2] = [-0.20, 0.50, 0.10, -0.10, 0.50, -0.20]  # Variance+burstiness
    W1[:, 3] = [0.10, 0.10, 0.10, 0.60, 0.10, 0.50]    # Punctuation+word-length
    B1 = np.zeros(64, dtype=np.float32)
    B1[0] = -0.15

    # Layer 2: 64 → 32
    W2 = (np.random.randn(64, 32) * np.sqrt(2.0 / 64)).astype(np.float32)
    W2[0, 0] = 0.50  # Primary signal path from weighted-avg neuron
    W2[1, 0] = 0.25
    W2[2, 0] = 0.25
    W2[3, 1] = 0.40
    B2 = np.zeros(32, dtype=np.float32)

    # Layer 3: 32 → 1
    W3 = (np.random.randn(32, 1) * np.sqrt(2.0 / 32)).astype(np.float32)
    W3[0, 0] = 0.70
    W3[1, 0] = 0.30
    B3 = np.array([0.0], dtype=np.float32)

    graph = onnx_pb2.GraphProto()
    graph.name = "text_classifier"

    # Initializers
    graph.initializer.append(_make_tensor("W1", W1))
    graph.initializer.append(_make_tensor("B1", B1))
    graph.initializer.append(_make_tensor("W2", W2))
    graph.initializer.append(_make_tensor("B2", B2))
    graph.initializer.append(_make_tensor("W3", W3))
    graph.initializer.append(_make_tensor("B3", B3))

    # Nodes
    graph.node.append(_make_node("MatMul", ["input", "W1"], ["mm1"], "matmul1"))
    graph.node.append(_make_node("Add", ["mm1", "B1"], ["fc1"], "add1"))
    graph.node.append(_make_node("Relu", ["fc1"], ["relu1"], "relu1"))
    graph.node.append(_make_node("MatMul", ["relu1", "W2"], ["mm2"], "matmul2"))
    graph.node.append(_make_node("Add", ["mm2", "B2"], ["fc2"], "add2"))
    graph.node.append(_make_node("Relu", ["fc2"], ["relu2"], "relu2"))
    graph.node.append(_make_node("MatMul", ["relu2", "W3"], ["mm3"], "matmul3"))
    graph.node.append(_make_node("Add", ["mm3", "B3"], ["logit"], "add3"))
    graph.node.append(_make_node("Sigmoid", ["logit"], ["output"], "sigmoid"))

    # IO
    graph.input.append(_make_value_info("input", TensorProto.FLOAT, [1, 6]))
    graph.output.append(_make_value_info("output", TensorProto.FLOAT, [1, 1]))

    _save_model(graph, str(dest))
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  NPU Content Analyzer — Model Setup")
    print("=" * 60)
    print()

    # Image model
    print("[1/2] Image model")
    have_mobilenet = download_image_model()
    if not have_mobilenet:
        create_custom_image_model()
    print()

    # Text model
    print("[2/2] Text classifier")
    create_text_model()
    print()

    # Verify with onnxruntime
    print("Verifying models with ONNX Runtime …")
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  Available providers: {providers}")

    for model_file in MODELS_DIR.glob("*.onnx"):
        try:
            sess = ort.InferenceSession(
                str(model_file),
                providers=providers,
            )
            inp = sess.get_inputs()[0]
            out = sess.get_outputs()[0]
            print(f"  ✓ {model_file.name}: input={inp.name} {inp.shape} → output={out.name} {out.shape} [{sess.get_providers()[0]}]")
        except Exception as e:
            print(f"  ✗ {model_file.name}: {e}")

    print()
    print("Setup complete.")


if __name__ == "__main__":
    main()
