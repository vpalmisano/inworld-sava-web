# Exporting SAVA

Activate your Python virtual environment, then install the dependencies using
the same interpreter that will run the exporter and tests:

```sh
python -m pip install inworld-sava onnx 'onnxruntime>=1.32' onnx-ir
```

Export the registered release, or pass a local SAVA safetensors checkpoint:

```sh
python export/export.py inworld/sava
```

This creates `models/sava.onnx` (FP32), `models/sava-q4.onnx` (4-bit
weight-only), `models/sava-q8.onnx` (dynamic 8-bit), and the shared
`models/sava.json` manifest. The exporter requires a checkpoint with three
lookahead frames. All variants take PCM and worklet-generated log-mel features;
the manifest describes their initially zeroed recurrent state. Quantization can
change predictions, and operator support depends on the browser and backend.
Models exported by other tools are not compatible with this package's worker
contract. Optional positional arguments override the FP32 model and manifest
paths; quantized variants are written alongside the FP32 model.

To verify export fidelity with a small random model:

```sh
python -m pip install pytest onnxruntime
python -m pytest -q export/test_export.py
```
