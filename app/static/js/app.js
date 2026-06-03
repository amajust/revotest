const API = '/api/v1/transcribe';

const $ = (id) => document.getElementById(id);
const el = {
  recordBtn: $('recordBtn'),
  recordLabel: $('recordLabel'),
  fileInput: $('fileInput'),
  statusText: $('statusText'),
  statusIndicator: $('statusIndicator'),
  recordingBar: $('recordingBar'),
  recordingTimer: $('recordingTimer'),
  waveform: $('waveform'),
  resultsPanel: $('resultsPanel'),
  rawResults: $('rawResults'),
  postResults: $('postResults'),
  tabs: document.querySelectorAll('.tab'),
};

let recorder = null;
let chunks = [];
let recording = false;
let timer = null;
let startTime = 0;

el.tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    el.tabs.forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');
    const target = tab.dataset.tab;
    el.rawResults.classList.toggle('hidden', target !== 'raw');
    el.postResults.classList.toggle('hidden', target !== 'post');
  });
});

el.fileInput.addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  await sendFile(f);
  el.fileInput.value = '';
});

el.recordBtn.addEventListener('click', () => {
  recording ? stop() : start();
});

function setStatus(text, type) {
  el.statusText.textContent = text;
  el.statusIndicator.className = 'status-indicator ' + type;
}

function setBusy(b) {
  el.recordBtn.disabled = b;
}

async function start() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('Mic not available', 'idle');
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';

    recorder = new MediaRecorder(stream, { mimeType: mime });
    chunks = [];

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      processBlob(new Blob(chunks, { type: mime }));
    };

    recorder.start(100);
    recording = true;
    startTime = Date.now();

    el.recordBtn.classList.add('recording');
    el.recordLabel.textContent = 'Stop';
    el.recordingBar.classList.remove('hidden');
    setStatus('Recording...', 'recording');
    startTimer();
    animateWave();
  } catch {
    setStatus('Mic access denied', 'idle');
  }
}

function stop() {
  if (!recorder || recorder.state === 'inactive') return;
  recorder.stop();
  recording = false;
  stopTimer();
  el.recordBtn.classList.remove('recording');
  el.recordLabel.textContent = 'Record';
  el.recordingBar.classList.add('hidden');
  setStatus('Processing...', 'processing');
  setBusy(true);
  el.waveform.innerHTML = '';
}

function startTimer() {
  stopTimer();
  timer = setInterval(() => {
    const sec = Math.floor((Date.now() - startTime) / 1000);
    el.recordingTimer.textContent =
      String(Math.floor(sec / 60)).padStart(2, '0') + ':' + String(sec % 60).padStart(2, '0');
  }, 200);
}

function stopTimer() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

function animateWave() {
  el.waveform.innerHTML = '';
  for (let i = 0; i < 30; i++) {
    const bar = document.createElement('div');
    bar.className = 'waveform-bar';
    bar.style.animationDelay = (i / 30) * 0.5 + 's';
    bar.style.height = 4 + Math.random() * 32 + 'px';
    el.waveform.appendChild(bar);
  }
}

async function processBlob(blob) {
  try {
    const buf = await blob.arrayBuffer();
    const ctx = new AudioContext({ sampleRate: 16000 });
    const decoded = await ctx.decodeAudioData(buf);
    const data = decoded.getChannelData(0);
    await ctx.close();

    const wav = WavEncoder.encode(data, decoded.sampleRate);
    await sendFile(new File([wav], 'recording.wav', { type: 'audio/wav' }));
  } catch {
    setStatus('Audio decode failed', 'idle');
    setBusy(false);
  }
}

async function sendFile(file) {
  setStatus('Transcribing...', 'processing');
  setBusy(true);

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch(API, { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    showResults(data);
    setStatus('Done', 'idle');
  } catch (err) {
    setStatus('Error: ' + err.message, 'idle');
  } finally {
    setBusy(false);
  }
}

function showResults(data) {
  el.resultsPanel.classList.remove('hidden');
  renderSegments(el.rawResults, data.raw_results, false);
  renderSegments(el.postResults, data.post_processed_results, true);
  el.tabs[0].click();
}

function renderSegments(container, segs, showConf) {
  if (!segs || !segs.length) {
    container.innerHTML = '<p class="placeholder">No segments.</p>';
    return;
  }
  container.innerHTML = '';
  for (const s of segs) {
    const div = document.createElement('div');
    div.className = 'segment';
    if (showConf && s.low_confidence) div.classList.add('low-confidence');

    const hdr = document.createElement('div');
    hdr.className = 'segment-header';

    const ts = document.createElement('span');
    ts.className = 'segment-timestamp';
    ts.textContent = s.start.toFixed(2) + 's \u2192 ' + s.end.toFixed(2) + 's';
    hdr.appendChild(ts);

    const lp = document.createElement('span');
    lp.className = 'segment-logprob';
    lp.textContent = 'logprob: ' + s.avg_logprob.toFixed(4);
    hdr.appendChild(lp);

    const txt = document.createElement('div');
    txt.className = 'segment-text';
    if (showConf && s.low_confidence) {
      const tag = document.createElement('span');
      tag.className = 'confidence-tag';
      tag.textContent = 'Low Confidence';
      txt.appendChild(tag);
    }
    txt.appendChild(document.createTextNode(s.text));
    div.appendChild(hdr);
    div.appendChild(txt);
    container.appendChild(div);
  }
}

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/sw.js').catch(() => {});
}

setStatus('Ready', 'idle');
