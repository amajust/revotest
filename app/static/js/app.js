(() => {
  'use strict';

  const API_URL = '/api/v1/transcribe';

  const elements = {
    recordBtn: document.getElementById('recordBtn'),
    recordLabel: document.getElementById('recordLabel'),
    fileInput: document.getElementById('fileInput'),
    statusText: document.getElementById('statusText'),
    statusIndicator: document.getElementById('statusIndicator'),
    statusBar: document.getElementById('statusBar'),
    recordingBar: document.getElementById('recordingBar'),
    recordingTimer: document.getElementById('recordingTimer'),
    waveform: document.getElementById('waveform'),
    resultsPanel: document.getElementById('resultsPanel'),
    rawResults: document.getElementById('rawResults'),
    postResults: document.getElementById('postResults'),
    tabs: document.querySelectorAll('.tab'),
  };

  let mediaRecorder = null;
  let audioChunks = [];
  let recordingStartTime = 0;
  let timerInterval = null;
  let isRecording = false;

  // Tab switching
  elements.tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      elements.tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.dataset.tab;
      elements.rawResults.classList.toggle('hidden', target !== 'raw');
      elements.postResults.classList.toggle('hidden', target !== 'post');
    });
  });

  // Upload handler
  elements.fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await processFile(file);
    elements.fileInput.value = '';
  });

  // Record button
  elements.recordBtn.addEventListener('click', () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  function setStatus(text, type) {
    elements.statusText.textContent = text;
    elements.statusIndicator.className = 'status-indicator ' + type;
  }

  function setBusy(busy) {
    elements.recordBtn.disabled = busy;
  }

  async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('Microphone not available in this browser', 'idle');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

      mediaRecorder = new MediaRecorder(stream, { mimeType });
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunks, { type: mimeType });
        processRecording(blob);
      };

      mediaRecorder.start(100);
      isRecording = true;
      recordingStartTime = Date.now();

      elements.recordBtn.classList.add('recording');
      elements.recordLabel.textContent = 'Stop';
      elements.recordingBar.classList.remove('hidden');
      setStatus('Recording...', 'recording');
      startTimer();
      animateWaveform();

    } catch (err) {
      setStatus('Microphone access denied', 'idle');
    }
  }

  function stopRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
    mediaRecorder.stop();
    isRecording = false;
    stopTimer();
    elements.recordBtn.classList.remove('recording');
    elements.recordLabel.textContent = 'Record';
    elements.recordingBar.classList.add('hidden');
    setStatus('Processing...', 'processing');
    setBusy(true);
    clearWaveform();
  }

  function startTimer() {
    stopTimer();
    timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
      const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const secs = String(elapsed % 60).padStart(2, '0');
      elements.recordingTimer.textContent = `${mins}:${secs}`;
    }, 200);
  }

  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  }

  function animateWaveform() {
    elements.waveform.innerHTML = '';
    const count = 30;
    for (let i = 0; i < count; i++) {
      const bar = document.createElement('div');
      bar.className = 'waveform-bar';
      const delay = (i / count) * 0.5;
      const height = 4 + Math.random() * 32;
      bar.style.animationDelay = `${delay}s`;
      bar.style.height = `${height}px`;
      elements.waveform.appendChild(bar);
    }
  }

  function clearWaveform() {
    elements.waveform.innerHTML = '';
  }

  async function processRecording(recordedBlob) {
    try {
      const arrayBuffer = await recordedBlob.arrayBuffer();
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const channelData = audioBuffer.getChannelData(0);
      await audioCtx.close();

      const wavBlob = WavEncoder.encode(channelData, audioBuffer.sampleRate);
      const file = new File([wavBlob], 'recording.wav', { type: 'audio/wav' });
      await processFile(file);
    } catch (err) {
      setStatus('Failed to decode audio', 'idle');
      setBusy(false);
    }
  }

  async function processFile(file) {
    setStatus('Transcribing...', 'processing');
    setBusy(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      displayResults(data);
      setStatus('Complete', 'idle');
    } catch (err) {
      setStatus(`Error: ${err.message}`, 'idle');
    } finally {
      setBusy(false);
    }
  }

  function displayResults(data) {
    elements.resultsPanel.classList.remove('hidden');
    renderSegments(elements.rawResults, data.raw_results, false);
    renderSegments(elements.postResults, data.post_processed_results, true);
    elements.tabs[0].click();
  }

  function renderSegments(container, segments, showConfidence) {
    if (!segments || segments.length === 0) {
      container.innerHTML = '<p class="placeholder">No segments returned.</p>';
      return;
    }

    container.innerHTML = '';
    for (const seg of segments) {
      const div = document.createElement('div');
      div.className = 'segment';
      if (showConfidence && seg.low_confidence) {
        div.classList.add('low-confidence');
      }

      const header = document.createElement('div');
      header.className = 'segment-header';

      const timestamp = document.createElement('span');
      timestamp.className = 'segment-timestamp';
      timestamp.textContent = `${seg.start.toFixed(2)}s → ${seg.end.toFixed(2)}s`;
      header.appendChild(timestamp);

      const logprob = document.createElement('span');
      logprob.className = 'segment-logprob';
      logprob.textContent = `logprob: ${seg.avg_logprob.toFixed(4)}`;
      header.appendChild(logprob);

      const textDiv = document.createElement('div');
      textDiv.className = 'segment-text';

      if (showConfidence && seg.low_confidence) {
        const tag = document.createElement('span');
        tag.className = 'confidence-tag';
        tag.textContent = 'Low Confidence';
        textDiv.appendChild(tag);
      }

      textDiv.appendChild(document.createTextNode(seg.text));
      div.appendChild(header);
      div.appendChild(textDiv);
      container.appendChild(div);
    }
  }

  // PWA: register service worker
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/sw.js').catch(() => {});
  }

  setStatus('Ready', 'idle');
})();
