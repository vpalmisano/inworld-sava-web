import json
import sys

import numpy as np
import onnx
import onnxruntime
import torch
from inworld.sava import config, model

from export import StreamingStep, checkpoint, flatten, main


def test_export_streaming_matches_reference(tmp_path, monkeypatch):
    attention = config.AttentionConfig(num_heads=2, num_kv_heads=1, head_dim=4,
                                       sliding_window=4)
    branch = config.BranchConfig(hidden_size=8, ffn_size=16, attention=attention)
    cfg = config.SavaConfig(hidden_size=8, ffn_size=16, layers=('mamba', 'attention'),
                            mamba=config.Mamba2Config(head_dim=4, state_size=4),
                            attention=attention, vad_branch=branch, turn_branch=branch)
    torch.manual_seed(42)
    net = model.SavaModel(cfg).eval()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['export.py', 'test-checkpoint'])
    monkeypatch.setattr(checkpoint, 'load', lambda source: net)
    main()
    path = tmp_path / 'models' / 'sava.onnx'
    manifest_path = tmp_path / 'models' / 'sava.json'
    manifest = json.loads(manifest_path.read_text())
    variants = [path, path.with_name('sava-q4.onnx'), path.with_name('sava-q8.onnx')]
    assert any(node.op_type == 'MatMulNBits' for node in onnx.load(variants[1]).graph.node)
    assert any(node.op_type == 'MatMulInteger' for node in onnx.load(variants[2]).graph.node)
    sessions = [onnxruntime.InferenceSession(str(variant), providers=['CPUExecutionProvider'])
                for variant in variants]
    states = [flatten(StreamingStep(net).template) for _ in variants]
    frontend = net.embedder.init_state(1, torch.device('cpu'))
    reference_state = net.init_state(1, torch.device('cpu'), torch.float32)
    rng = np.random.default_rng(123)
    for index in range(6):
        samples = torch.from_numpy(rng.normal(0, 0.1, (1, 1, 640)).astype('float32'))
        mel, frontend = net.embedder.forward_chunk(samples, frontend)
        with torch.inference_mode():
            reference, reference_state = net.step(samples.reshape(1, 640), reference_state)
        expected = [torch.sigmoid(reference.lookahead), torch.sigmoid(reference.speech),
                    torch.sigmoid(reference.turn[:, :, :1])]
        for variant_index, session in enumerate(sessions):
            feeds = {'samples': samples.numpy(), 'mel': mel.numpy()}
            feeds.update({spec['input']: tensor.detach().numpy()
                          for spec, tensor in zip(manifest['states'], states[variant_index], strict=True)})
            actual = session.run(None, feeds)
            tolerance = 1e-5 if variant_index == 0 else 0.05
            for observed, wanted in zip(actual[:3], expected, strict=True):
                np.testing.assert_allclose(observed, wanted.numpy(), atol=tolerance, rtol=0,
                                           err_msg=f'{variants[variant_index].name}, frame {index}')
            states[variant_index] = [torch.from_numpy(value) for value in actual[3:]]