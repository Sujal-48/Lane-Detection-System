(function () {
  const fileInput = document.getElementById('fileInput');
  const uploadBtn = document.getElementById('uploadBtn');
  const video = document.getElementById('video');
  const imagePreview = document.getElementById('imagePreview');
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  const emptyState = document.getElementById('emptyState');
  const playBtn = document.getElementById('playBtn');
  const resetBtn = document.getElementById('resetBtn');
  const detectBtn = document.getElementById('detectBtn');
  const overlayToggle = document.getElementById('overlayToggle');
  const statusPill = document.getElementById('statusPill');
  const statusText = document.getElementById('statusText');
  const mStatus = document.getElementById('mStatus');
  const mLeft = document.getElementById('mLeft');
  const mRight = document.getElementById('mRight');
  const mCurve = document.getElementById('mCurve');
  const mOffset = document.getElementById('mOffset');
  const offsetMarker = document.getElementById('offsetMarker');
  const pipelineItems = document.querySelectorAll('.pipeline-item');
  const logBox = document.getElementById('logBox');

  // The Flask server runs on the same computer and exposes the trained model.
  const API_URL = 'http://localhost:5000';
  let backendAvailable = false;
  const sendCanvas = document.createElement('canvas');
  const sendCtx = sendCanvas.getContext('2d');

  let detecting = false;
  let rafId = null;
  let pipelineTimer = null;
  let sendTimer = null;
  let requestInFlight = false;
  const startTime = Date.now();

  function log(msg) {
    const t = ((Date.now() - startTime) / 1000).toFixed(1).padStart(5, '0');
    const line = document.createElement('div');
    line.className = 'log-line';
    line.innerHTML = '<span>' + t + 's</span>' + msg;
    logBox.appendChild(line);
    logBox.scrollTop = logBox.scrollHeight;
  }

  function resizeCanvas() {
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
  }
  window.addEventListener('resize', resizeCanvas);

  // The dashboard remains usable as a visual demo if the Python server is off.
  fetch(API_URL + '/health')
    .then(r => r.ok ? r.json() : Promise.reject())
    .then(data => {
      backendAvailable = true;
      log('connected to ML backend (' + data.device + ')');
    })
    .catch(() => log('ML backend is offline — demo mode is available'));

  uploadBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);

  emptyState.style.display = 'none';
  resetBtn.disabled = false;
  detectBtn.disabled = false;
  log('loaded file "' + file.name + '"');

  if (file.type.startsWith('image/')) {
    video.pause();
    video.removeAttribute('src');
    video.load();

    imagePreview.src = url;
    imagePreview.hidden = false;
    imagePreview.onload = resizeCanvas;

    playBtn.disabled = true;
    statusText.textContent = 'Image loaded — ready for detection';
  } else {
    imagePreview.hidden = true;
    imagePreview.removeAttribute('src');

    video.src = url;
    video.load();
    video.addEventListener('loadedmetadata', resizeCanvas, { once: true });

    playBtn.disabled = false;
    statusText.textContent = 'Video loaded — ready';
  }
});

  playBtn.addEventListener('click', () => {
    if (video.paused) {
      video.play();
      playBtn.textContent = 'Pause';
    } else {
      video.pause();
      playBtn.textContent = 'Play';
    }
  });

  resetBtn.addEventListener('click', () => {
    stopDetection();
    video.pause();
    video.currentTime = 0;
    video.removeAttribute('src');
    video.load();
    emptyState.style.display = 'flex';
    playBtn.disabled = true;
    resetBtn.disabled = true;
    detectBtn.disabled = true;
    playBtn.textContent = 'Play';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    statusText.textContent = 'Idle — no feed loaded';
    statusPill.classList.remove('live');
    mStatus.textContent = 'Standby';
    mLeft.textContent = '—';
    mRight.textContent = '—';
    mCurve.textContent = '—';
    mOffset.textContent = '—';
    offsetMarker.style.left = '50%';
    pipelineItems.forEach(p => p.classList.remove('active', 'done'));
    log('session reset');
  });

  detectBtn.addEventListener('click', () => {
    if (!detecting) startDetection(); else stopDetection();
  });

  function startDetection() {
  detecting = true;
  detectBtn.textContent = 'Stop detection';
  statusPill.classList.add('live');
  mStatus.textContent = 'Running';
  runPipelineCycle();

  if (backendAvailable) {
    statusText.textContent = 'Detection running — live model';

    if (!imagePreview.hidden) {
      log('image sent to ML backend');
      sendImageToBackend();
    } else {
      log('detection started — sending video frames to ML backend');
      sendTimer = setInterval(sendFrameToBackend, 250);
    }
  } else {
    statusText.textContent = 'Detection running — demo simulation';
    log('ML backend is offline');
    drawDemoFrame();
  }
}

  function stopDetection() {
    detecting = false;
    detectBtn.textContent = 'Start detection';
    statusPill.classList.remove('live');
    statusText.textContent = video.src ? 'Feed loaded — ready' : 'Idle — no feed loaded';
    mStatus.textContent = 'Standby';
    if (rafId) cancelAnimationFrame(rafId);
    if (pipelineTimer) clearInterval(pipelineTimer);
    if (sendTimer) clearInterval(sendTimer);
    pipelineItems.forEach(p => p.classList.remove('active', 'done'));
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    log('detection stopped');
  }

  function runPipelineCycle() {
    let step = 0;
    pipelineTimer = setInterval(() => {
      pipelineItems.forEach((p, i) => {
        p.classList.toggle('active', i === step);
        p.classList.toggle('done', i < step);
      });
      step = (step + 1) % (pipelineItems.length + 1);
      if (step === pipelineItems.length) {
        pipelineItems.forEach(p => p.classList.add('done'));
      }
    }, 480);
  }
  function sendImageToBackend() {
  if (!detecting || imagePreview.hidden || requestInFlight) return;

  sendCanvas.width = imagePreview.naturalWidth;
  sendCanvas.height = imagePreview.naturalHeight;
  sendCtx.drawImage(imagePreview, 0, 0, sendCanvas.width, sendCanvas.height);

  const dataUrl = sendCanvas.toDataURL('image/jpeg', 0.9);

  requestInFlight = true;
  fetch(API_URL + '/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataUrl }),
  })
    .then(r => r.ok ? r.json() : r.json().then(data => Promise.reject(data)))
    .then(data => {
      requestInFlight = false;
      renderPrediction(data);
      log('image detection complete');
    })
    .catch(() => {
      requestInFlight = false;
      log('image detection failed — check Python server');
    });
  }

  function sendFrameToBackend() {
    if (!detecting || video.paused || video.readyState < 2 || requestInFlight) return;

    sendCanvas.width = video.videoWidth;
    sendCanvas.height = video.videoHeight;
    sendCtx.drawImage(video, 0, 0, sendCanvas.width, sendCanvas.height);

    requestInFlight = true;
    fetch(API_URL + '/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: sendCanvas.toDataURL('image/jpeg', 0.7) }),
    })
      .then(r => r.ok ? r.json() : r.json().then(data => Promise.reject(data)))
      .then(data => {
        requestInFlight = false;
        renderPrediction(data);
      })
      .catch(error => {
        requestInFlight = false;
        backendAvailable = false;
        clearInterval(sendTimer);
        log('backend error — switching to demo simulation');
        drawDemoFrame();
      });
  }

  function renderPrediction(data) {
    resizeCanvas();
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    mCurve.textContent = data.curvature.toFixed(1);
    mLeft.textContent = data.coverage.toFixed(1) + '%';
    mRight.textContent = data.coverage.toFixed(1) + '%';
    mOffset.textContent = (data.offset_pct > 0 ? '+' : '') + data.offset_pct.toFixed(1) + '%';
    offsetMarker.style.left = (50 + Math.max(-50, Math.min(50, data.offset_pct)) / 2) + '%';

    if (overlayToggle.checked && data.mask) {
      const image = new Image();
      image.onload = () => ctx.drawImage(image, 0, 0, w, h);
      image.src = data.mask;
    }
  }

  // Demo simulation is deliberately retained when no trained model is running.
  function drawDemoFrame() {
    if (!detecting) return;
    resizeCanvas();
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const t = Date.now() / 900;
    const wobble = Math.sin(t) * 0.06;
    const curveVal = (Math.sin(t * 0.5) * 480 + 900).toFixed(0);
    const leftConf = (86 + Math.sin(t * 1.3) * 8).toFixed(1);
    const rightConf = (89 + Math.cos(t * 1.1) * 7).toFixed(1);
    const offsetVal = (wobble * 100).toFixed(1);

    mCurve.textContent = curveVal + ' m';
    mLeft.textContent = leftConf + '%';
    mRight.textContent = rightConf + '%';
    mOffset.textContent = (offsetVal > 0 ? '+' : '') + offsetVal + ' cm';
    offsetMarker.style.left = (50 + wobble * 220) + '%';

    if (overlayToggle.checked) {
      const baseL = w * (0.32 + wobble);
      const baseR = w * (0.68 + wobble);
      const topL = w * (0.46 + wobble * 0.4);
      const topR = w * (0.54 + wobble * 0.4);
      const topY = h * 0.38, botY = h;

      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.strokeStyle = '#F5C542';
      ctx.beginPath();
      ctx.moveTo(baseL, botY);
      ctx.lineTo(topL, topY);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(baseR, botY);
      ctx.lineTo(topR, topY);
      ctx.stroke();

      ctx.fillStyle = 'rgba(63, 217, 208, 0.14)';
      ctx.beginPath();
      ctx.moveTo(baseL, botY);
      ctx.lineTo(topL, topY);
      ctx.lineTo(topR, topY);
      ctx.lineTo(baseR, botY);
      ctx.closePath();
      ctx.fill();
    }

    rafId = requestAnimationFrame(drawDemoFrame);
  }
})();
