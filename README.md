# inworld-sava-web

[GitHub](https://github.com/vpalmisano/inworld-sava-web/) | [Hugging Face Model](https://huggingface.co/vpalmisano/inworld-sava-web) | [Demo](https://vpalmisano.github.io/inworld-sava-web/)

Browser streaming voice activity and conversational end-of-turn detection using
[Inworld SAVA](https://huggingface.co/inworld/sava) model.

## Build and use

```sh
npm install
npm test
```

Run `npm run demo` and open http://localhost:5173 to try the demo. The demo can select
the FP32, Q4, or Q8 model variants on WASM or WebGPU backends.
The demo loads Vue, Vuetify, and models from a CDN and requires network access.

Example usage:

```ts
import { SavaDetector } from '@vpalmisano/inworld-sava-web';

const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const detector = new SavaDetector({
  modelUrl: 'https://huggingface.co/vpalmisano/inworld-sava-web/resolve/main/sava.onnx',
  manifestUrl: 'https://huggingface.co/vpalmisano/inworld-sava-web/resolve/main/sava.json',
  backend: 'auto',
  onSpeechStart: frame => console.log('speech', frame),
  onTurn: turn => console.log('turn', turn),
  onError: error => console.error(error),
});
await detector.start(stream.getAudioTracks()[0]);
// Later, flush before releasing the caller-owned track:
await detector.stop();
stream.getTracks().forEach(track => track.stop());
```

Refer to [export](/export/README.md) for instructions to export SAVA from the Inworld AI repository.

## Attribution

SAVA and its reference code and weights are copyright Inworld AI and licensed under [Apache-2.0](https://github.com/inworld-ai/sava/blob/main/LICENSE). The frontend and turn policy follow the upstream implementation. Preserve the upstream license and notices when redistributing SAVA assets.
