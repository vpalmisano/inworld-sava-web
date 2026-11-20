import { DEFAULT_POLICY, SavaDetector } from './dist/index.js';

const { createApp, ref, computed, watch, onMounted, onBeforeUnmount } = globalThis.Vue;

const { createVuetify } = globalThis.Vuetify;

const vuetify = createVuetify({
  theme: {
    defaultTheme: 'light',
    themes: { light: { colors: { primary: '#176c54', warning: '#bd6845' } } },
  },
});

createApp({
  setup() {
    const modelUrl = ref(
      'https://huggingface.co/vpalmisano/inworld-sava-web/resolve/main/sava.onnx',
    );

    const manifestUrl = ref(
      'https://huggingface.co/vpalmisano/inworld-sava-web/resolve/main/sava.json',
    );

    const modelVariant = ref(modelUrl.value);

    const backend = ref('auto');

    const policy = ref({ ...DEFAULT_POLICY });

    const modelVariants = [
      {
        title: 'FP32',
        value: 'https://huggingface.co/vpalmisano/inworld-sava-web/resolve/main/sava.onnx',
      },
      {
        title: 'Q4',
        value: 'https://huggingface.co/vpalmisano/inworld-sava-web/resolve/main/sava-q4.onnx',
      },
      {
        title: 'Q8',
        value: 'https://huggingface.co/vpalmisano/inworld-sava-web/resolve/main/sava-q8.onnx',
      },
      { title: 'Custom URL', value: 'custom' },
    ];

    const backends = [
      { title: 'Automatic', value: 'auto' },
      { title: 'WASM', value: 'wasm' },
      { title: 'WebGPU', value: 'webgpu' },
    ];

    const phase = ref('idle');

    const error = ref('');

    const detection = ref('Standby');

    const activeBackend = ref('');

    const frameCount = ref(0);

    const speech = ref(null);

    const turnScore = ref(null);

    const starts = ref(0);

    const kept = ref(0);

    const turns = ref([]);

    const chart = ref(null);

    const history = [];

    const maxFrames = 1000;

    let detector = null;

    let stream = null;

    let resizeObserver = null;

    let animation = 0;

    let nextTurnId = 0;

    const busy = computed(() => phase.value !== 'idle');

    const statusText = computed(
      () =>
        ({ idle: 'Idle', starting: 'Starting', running: 'Listening', stopping: 'Finishing' })[
          phase.value
        ],
    );

    const speechPercent = computed(() =>
      speech.value === null ? '—' : Math.round(speech.value * 100),
    );

    const turnPercent = computed(() =>
      turnScore.value === null ? '—' : Math.round(turnScore.value * 100),
    );

    const elapsed = computed(() => {
      const seconds = frameCount.value * 0.04;

      return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${(seconds % 60).toFixed(1).padStart(4, '0')}`;
    });

    watch(modelUrl, value => {
      modelVariant.value = modelVariants.some(item => item.value === value) ? value : 'custom';
    });
    function selectVariant(value) {
      if (value !== 'custom') modelUrl.value = value;
    }

    function drawChart() {
      const canvas = chart.value;

      if (!canvas) return;
      const bounds = canvas.getBoundingClientRect();

      const ratio = window.devicePixelRatio || 1;

      canvas.width = Math.max(1, Math.round(bounds.width * ratio));
      canvas.height = Math.max(1, Math.round(bounds.height * ratio));
      const context = canvas.getContext('2d');

      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      const { width, height } = bounds;

      context.clearRect(0, 0, width, height);
      context.strokeStyle = '#eaf0ea';
      context.lineWidth = 1;
      for (let index = 0; index <= 4; index++) {
        const y = Math.min(height - 0.5, (index * height) / 4 + 0.5);

        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
      }
      for (let index = 1; index < 8; index++) {
        const x = (index * width) / 8;

        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.stroke();
      }
      for (const [key, color] of [
        ['speech', '#188462'],
        ['turn', '#e78259'],
      ]) {
        if (!history.length) continue;
        context.beginPath();
        history.forEach((frame, index) => {
          const x = width - ((history.length - 1 - index) * width) / (maxFrames - 1);

          const y = height - Math.max(0, Math.min(1, frame[key])) * (height - 2) - 1;

          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.lineJoin = 'round';
        context.stroke();
      }
    }

    function scheduleChart() {
      if (animation) return;
      animation = requestAnimationFrame(() => {
        animation = 0;
        drawChart();
      });
    }

    function resetDisplay() {
      history.length = 0;
      frameCount.value = 0;
      speech.value = null;
      turnScore.value = null;
      starts.value = 0;
      kept.value = 0;
      turns.value = [];
      detection.value = 'Standby';
      activeBackend.value = '';
      scheduleChart();
    }

    function release() {
      const currentStream = stream;

      stream = null;
      detector = null;
      phase.value = 'idle';
      currentStream?.getTracks().forEach(track => track.stop());
    }

    function reportError(cause) {
      const message = cause instanceof Error ? cause.message : String(cause);

      if (message.includes('SAVA inference is slow')) {
        const usingQ8Wasm = modelUrl.value.endsWith('sava-q8.onnx') && backend.value === 'wasm';

        const hint = usingQ8Wasm
          ? 'Try WebGPU or close other applications to free resources.'
          : 'Try the F32 model with the WASM backend; Q8/Q4 can be slower on some devices.';

        error.value = `${message}. ${hint}`;
      } else error.value = message;
      detection.value = 'Stopped';
      release();
    }

    async function startSession() {
      if (busy.value) return;
      error.value = '';
      resetDisplay();
      phase.value = 'starting';
      try {
        if (!modelUrl.value.trim() || !manifestUrl.value.trim())
          throw new Error('Model and manifest URLs are required.');
        if (!navigator.mediaDevices?.getUserMedia)
          throw new Error('Microphone access requires a secure context.');
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const track = stream.getAudioTracks()[0];

        if (!track) throw new Error('No audio track was returned by the microphone.');
        const instance = new SavaDetector({
          modelUrl: new URL(modelUrl.value, location.href).href,
          manifestUrl: new URL(manifestUrl.value, location.href).href,
          backend: backend.value,
          policy: { ...policy.value },
          onFrame: frame => {
            history.push(frame);
            if (history.length > maxFrames) history.shift();
            speech.value = frame.speech;
            turnScore.value = frame.turn;
            frameCount.value = frame.index + 1;
            scheduleChart();
          },
          onSpeechStart: () => {
            starts.value++;
            detection.value = 'Speech';
          },
          onTurn: turn => {
            if (turn.kept) kept.value++;
            detection.value = 'Listening';
            turns.value.unshift({ ...turn, id: nextTurnId++ });
          },
          onError: reportError,
        });

        detector = instance;
        await instance.start(track);
        track.addEventListener(
          'ended',
          () => {
            if (detector === instance && phase.value === 'running') void stopSession();
          },
          { once: true },
        );
        activeBackend.value = instance.backend.toUpperCase();
        detection.value = 'Listening';
        phase.value = 'running';
      } catch (cause) {
        reportError(cause);
      }
    }

    async function stopSession() {
      if (!detector || phase.value !== 'running') return;
      phase.value = 'stopping';
      try {
        await detector.stop();
        release();
        detection.value = 'Stopped';
      } catch (cause) {
        reportError(cause);
      }
    }

    onMounted(() => {
      resizeObserver = new ResizeObserver(scheduleChart);
      resizeObserver.observe(chart.value);
      scheduleChart();
    });
    onBeforeUnmount(() => {
      resizeObserver?.disconnect();
      if (animation) cancelAnimationFrame(animation);
      void detector?.stop().catch(() => {});
      release();
    });
    return {
      modelUrl,
      manifestUrl,
      modelVariant,
      backend,
      policy,
      modelVariants,
      backends,
      phase,
      busy,
      error,
      detection,
      activeBackend,
      frameCount,
      starts,
      kept,
      turns,
      chart,
      statusText,
      speechPercent,
      turnPercent,
      elapsed,
      selectVariant,
      startSession,
      stopSession,
    };
  },
})
  .use(vuetify)
  .mount('#app');
