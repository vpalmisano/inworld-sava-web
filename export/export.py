"""Export an Inworld SAVA checkpoint as a single-frame streaming ONNX model."""

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from inworld.sava import checkpoint, model as model_lib


def flatten(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if dataclasses.is_dataclass(value):
        return [tensor for field in dataclasses.fields(value)
                for tensor in flatten(getattr(value, field.name))]
    if isinstance(value, list):
        return [tensor for item in value for tensor in flatten(item)]
    return []


def restore(template, tensors):
    if isinstance(template, torch.Tensor):
        return next(tensors)
    if dataclasses.is_dataclass(template):
        return dataclasses.replace(template, **{
            field.name: restore(getattr(template, field.name), tensors)
            for field in dataclasses.fields(template)
        })
    if isinstance(template, list):
        return [restore(item, tensors) for item in template]
    return template


class StreamingStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.template = dataclasses.replace(
            model.init_state(1, torch.device('cpu'), torch.float32), frontend=None)

    def forward(self, samples, mel, *tensors):
        state = restore(self.template, iter(tensors))
        hidden, mixer = self.model.mixer.forward_chunk(
            self.model.mixer.rows(mel, samples), state.mixer)
        trunk = []
        for block, block_state in zip(self.model.trunk, state.trunk, strict=True):
            hidden, block_state = block.forward_chunk(hidden, block_state)
            trunk.append(block_state)
        speech, lookahead, turn, vad, turn_state, prev = self.model._heads(
            self.model.final_norm(hidden), state.vad, state.turn, state.prev_speech)
        lookahead = model_lib._mask_lookahead(lookahead, state.frames, 3)
        next_state = dataclasses.replace(state, mixer=mixer, trunk=trunk, vad=vad,
                                         turn=turn_state, prev_speech=prev,
                                         frames=state.frames + 1)
        return (torch.sigmoid(lookahead), torch.sigmoid(speech),
                torch.sigmoid(turn[:, :, :1]), *flatten(next_state))


def export(model, model_path, manifest_path):
    if model.lookahead_frames != 3:
        raise ValueError('This package requires a model with three lookahead frames')
    wrapper = StreamingStep(model.eval())
    states = flatten(wrapper.template)
    names = [f'state_{index}' for index in range(len(states))]
    outputs = [f'next_state_{index}' for index in range(len(states))]
    manifest = {'format': 'sava-streaming-onnx-v1', 'states': [
        {'input': name, 'output': output, 'shape': list(tensor.shape),
         'dtype': 'int64' if tensor.dtype == torch.int64 else 'float32'}
        for name, output, tensor in zip(names, outputs, states, strict=True)
    ]}
    with torch.inference_mode():
        torch.onnx.export(wrapper, (torch.zeros(1, 1, 640), torch.zeros(1, 1, 512), *states),
                          str(model_path), input_names=['samples', 'mel', *names],
                          output_names=['speech', 'speech_on_time', 'turn', *outputs],
                          opset_version=17, dynamo=False)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def quantize_models(model_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    model_path = Path(model_path)
    q4_path = model_path.with_name(f'{model_path.stem}-q4.onnx')
    q8_path = model_path.with_name(f'{model_path.stem}-q8.onnx')
    quantizer = MatMulNBitsQuantizer(str(model_path), bits=4, block_size=32,
                                     is_symmetric=True, op_types_to_quantize=('MatMul',))
    quantizer.process()
    quantizer.model.save_model_to_file(str(q4_path))
    quantize_dynamic(model_path, q8_path, op_types_to_quantize=['MatMul'],
                     weight_type=QuantType.QInt8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='SAVA checkpoint path or registered model source')
    parser.add_argument('model', nargs='?', default='models/sava.onnx', help='output ONNX path')
    parser.add_argument('manifest', nargs='?', default='models/sava.json', help='output JSON manifest path')
    args = parser.parse_args()
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    export(checkpoint.load(args.checkpoint), args.model, args.manifest)
    quantize_models(args.model)


if __name__ == '__main__':
    main()