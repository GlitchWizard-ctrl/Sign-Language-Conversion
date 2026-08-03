// Live SignCast client script
const authToken = localStorage.getItem('authToken');
if (!authToken) {
  window.location.href = 'login.html';
}

const socket = io({ autoConnect: false });
let localStream = null;
let peerConnection = null;
let dataChannel = null;
let roomId = null;
let isCaller = false;
let isCameraOn = true;
let isMicOn = true;
let converterEnabled = true;
let lastPredictionTime = 0;
const MIN_PREDICT_INTERVAL = 200;

// ---- Audio (text-to-speech) state ----
let audioEnabled = true;
let lastSpokenSign = null;
let lastSpokenTime = 0;
const SPEAK_COOLDOWN_MS = 1800;   // don't repeat the same sign faster than this
const MIN_CONFIDENCE_TO_SPEAK = 0.6;

function speakSign(text) {
  if (!audioEnabled || !('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();  // stop any overlapping utterance
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const localOverlay = document.getElementById('localOverlay');
const remoteOverlay = document.getElementById('remoteOverlay');
const connectionStatus = document.getElementById('connectionStatus');
const activeRoomLabel = document.getElementById('activeRoomLabel');
const signLabelValue = document.getElementById('signLabelValue');
const confidenceValue = document.getElementById('confidenceValue');
const frameRateValue = document.getElementById('frameRateValue');

const roomIdInput = document.getElementById('roomIdInput');
const startRoomBtn = document.getElementById('startRoomBtn');
const joinRoomBtn = document.getElementById('joinRoomBtn');
const toggleCamBtn = document.getElementById('toggleCamBtn');
const toggleMicBtn = document.getElementById('toggleMicBtn');
const endCallBtn = document.getElementById('endCallBtn');
// in-call-only UI panels and overlay controls
const callActionsPanel = document.getElementById('callActionsPanel');
const callOverlayControls = document.getElementById('callOverlayControls');
const overlayToggleCam = document.getElementById('overlayToggleCam');
const overlayToggleMic = document.getElementById('overlayToggleMic');
const overlayInterpreterBtn = document.getElementById('overlayInterpreterBtn');
const overlayFullscreenBtn = document.getElementById('overlayFullscreenBtn');
const overlayEndCallBtn = document.getElementById('overlayEndCallBtn');
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');
const converterSwitch = document.getElementById('converterSwitch');
const converterLabel = document.getElementById('converterLabel');
// audio (text-to-speech) toggle — optional, only wired up if present in HTML
const audioSwitch = document.getElementById('audioSwitch');
const audioLabel = document.getElementById('audioLabel');

const toastContainer = document.getElementById('toastContainer');
const usernameLabel = document.getElementById('usernameLabel');
const userRoleLabel = document.getElementById('userRoleLabel');
const adminBanner = document.getElementById('adminBanner');
const signOutBtn = document.getElementById('signOutBtn');
const historyBody = document.getElementById('historyBody');
// user dropdown elements
const userPillBtn = document.getElementById('userPillBtn');
const userMenu = document.getElementById('userMenu');
const openHistory = document.getElementById('openHistory');

if (userPillBtn && userMenu) {
  userPillBtn.addEventListener('click', () => {
    const showing = userMenu.classList.toggle('show');
    userMenu.setAttribute('aria-hidden', !showing);
  });
  document.addEventListener('click', (e) => {
    if (!userMenu.contains(e.target) && !userPillBtn.contains(e.target)) {
      userMenu.classList.remove('show'); userMenu.setAttribute('aria-hidden', 'true');
    }
  });
}

if (openHistory) openHistory.addEventListener('click', (e) => { e.preventDefault(); location.href = 'history.html'; userMenu.classList.remove('show'); });

usernameLabel.textContent = localStorage.getItem('fullname') || localStorage.getItem('username') || 'Guest';
userRoleLabel.textContent = (localStorage.getItem('role') || 'user').toUpperCase();
adminBanner.style.display = 'none';

signOutBtn.addEventListener('click', async () => {
  await fetch('/api/logout', {
    method: 'POST',
    headers: { 'Authorization': authToken }
  }).catch(() => null);
  localStorage.removeItem('authToken');
  localStorage.removeItem('username');
  localStorage.removeItem('fullname');
  localStorage.removeItem('role');
  document.cookie = 'authToken=; path=/; max-age=0';
  window.location.href = 'login.html';
});

// DEBUG: log presence of key buttons so we can tell if handlers attach
console.log('UI elements:', { startRoomBtn: !!startRoomBtn, joinRoomBtn: !!joinRoomBtn, roomIdInput: !!roomIdInput });

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  const icon = type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle';
  toast.innerHTML = `<i class="fa-solid ${icon}"></i><span>${message}</span>`;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

function updateConnectionState(connected) {
  connectionStatus.textContent = connected ? 'Connected' : 'Disconnected';
  connectionStatus.className = connected ? 'status-pill status-pill-ready' : 'status-pill status-pill-disconnected';
  document.getElementById('connectionLabel').textContent = connected ? 'In call' : 'Disconnected';
}

function updateConverterLabel() {
  if (converterLabel) converterLabel.textContent = converterEnabled ? 'On' : 'Off';
}

function updateAudioLabel() {
  if (audioLabel) audioLabel.textContent = audioEnabled ? 'On' : 'Off';
}

function generateRoomCode() {
  return Math.random().toString(36).substring(2, 8).toUpperCase();
}

if (converterSwitch) {
  converterSwitch.addEventListener('change', () => {
    converterEnabled = converterSwitch.checked;
    updateConverterLabel();
    showToast(converterEnabled ? 'Live converter enabled' : 'Live converter disabled', 'info');
  });
}

if (audioSwitch) {
  audioSwitch.addEventListener('change', () => {
    audioEnabled = audioSwitch.checked;
    updateAudioLabel();
    if (!audioEnabled && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    showToast(audioEnabled ? 'Audio output enabled' : 'Audio output disabled', 'info');
  });
}

async function verifyProfile() {
  try {
    const response = await fetch('/api/profile', {
      headers: { 'Authorization': authToken }
    });
    const data = await response.json();
    if (!response.ok || !data.success) {
      throw new Error('Unauthorized');
    }

    usernameLabel.textContent = data.fullname || data.username || 'Guest';
    userRoleLabel.textContent = (data.role || 'user').toUpperCase();

    if (data.role === 'admin') {
      adminBanner.style.display = 'flex';
      localStorage.setItem('role', 'admin');
    } else {
      adminBanner.style.display = 'none';
      localStorage.setItem('role', 'user');
    }
    return true;
  } catch (err) {
    console.error('Profile verification failed:', err);
    localStorage.removeItem('authToken');
    localStorage.removeItem('username');
    localStorage.removeItem('fullname');
    localStorage.removeItem('role');
    window.location.href = 'login.html';
    return false;
  }
}

async function fetchCallHistory() {
  try {
    const response = await fetch('/api/call-history', {
      headers: { 'Authorization': authToken }
    });
    const data = await response.json();
    if (!data.success) throw new Error('Unable to load history');
    renderHistory(data.history || []);
  } catch (err) {
    console.error(err);
    showToast('Could not load call history.', 'error');
  }
}

function renderHistory(history) {
  if (!history.length) {
    historyBody.innerHTML = '<tr><td colspan="5" class="history-empty">No recent calls yet.</td></tr>';
    return;
  }
  historyBody.innerHTML = history.map(item => {
    return `<tr>
      <td>${item.room_id}</td>
      <td>${item.caller}</td>
      <td>${item.callee}</td>
      <td>${item.interpreter_mode ? 'Yes' : 'No'}</td>
      <td>${item.duration_seconds}s</td>
    </tr>`;
  }).join('');
}

async function startLocalMedia() {
  if (localStream) return localStream;
  try {
    localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    localVideo.srcObject = localStream;
    // Set initial button icons and states (guard in case top-level controls were removed)
    if (toggleCamBtn) toggleCamBtn.innerHTML = '<i class="fa-solid fa-video"></i> Camera ON';
    if (toggleMicBtn) toggleMicBtn.innerHTML = '<i class="fa-solid fa-microphone"></i> Mic ON';
    if (overlayToggleCam) overlayToggleCam.innerHTML = '<i class="fa-solid fa-video"></i>';
    if (overlayToggleMic) overlayToggleMic.innerHTML = '<i class="fa-solid fa-microphone"></i>';
    if (converterSwitch) converterSwitch.checked = converterEnabled;
    if (audioSwitch) audioSwitch.checked = audioEnabled;
    updateConverterLabel();
    updateAudioLabel();
    isCameraOn = true;
    isMicOn = true;
    return localStream;
  } catch (err) {
    console.error(err);
    showToast('Unable to access camera or microphone.', 'error');
    return null;
  }
}

function toggleCamera() {
  if (!localStream) return;
  const videoTrack = localStream.getVideoTracks()[0];
  if (!videoTrack) return;
  videoTrack.enabled = !videoTrack.enabled;
  isCameraOn = videoTrack.enabled;
  // Use icon + status
  if (toggleCamBtn) toggleCamBtn.innerHTML = isCameraOn ? '<i class="fa-solid fa-video"></i> Camera ON' : '<i class="fa-solid fa-video-slash"></i> Camera OFF';
  if (overlayToggleCam) overlayToggleCam.innerHTML = isCameraOn ? '<i class="fa-solid fa-video"></i>' : '<i class="fa-solid fa-video-slash"></i>';
  localVideo.classList.toggle('video-muted', !isCameraOn);
  localVideo.style.filter = isCameraOn ? 'none' : 'brightness(0.02)';
  localVideo.style.opacity = isCameraOn ? '1' : '0.1';
  localVideo.style.backgroundColor = isCameraOn ? 'transparent' : '#000';
  const localHint = document.getElementById('localCameraHint');
  if (localHint) localHint.style.display = isCameraOn ? 'none' : 'flex';
  if (!isCameraOn) {
    localOverlay.style.display = 'none';
  } else {
    localOverlay.style.display = 'block';
  }
  // notify other participants
  if (socket && socket.connected && roomId) {
    socket.emit('camera-toggle', { room_id: roomId, enabled: isCameraOn });
  }
}

function toggleMic() {
  if (!localStream) return;
  const audioTrack = localStream.getAudioTracks()[0];
  if (!audioTrack) return;
  audioTrack.enabled = !audioTrack.enabled;
  isMicOn = audioTrack.enabled;
  if (toggleMicBtn) toggleMicBtn.innerHTML = isMicOn ? '<i class="fa-solid fa-microphone"></i> Mic ON' : '<i class="fa-solid fa-microphone-slash"></i> Mic OFF';
  if (overlayToggleMic) overlayToggleMic.innerHTML = isMicOn ? '<i class="fa-solid fa-microphone"></i>' : '<i class="fa-solid fa-microphone-slash"></i>';
  if (socket && socket.connected && roomId) {
    socket.emit('mic-toggle', { room_id: roomId, enabled: isMicOn });
  }
}

if (toggleCamBtn) toggleCamBtn.addEventListener('click', toggleCamera);
if (toggleMicBtn) toggleMicBtn.addEventListener('click', toggleMic);
if (overlayToggleCam) overlayToggleCam.addEventListener('click', toggleCamera);
if (overlayToggleMic) overlayToggleMic.addEventListener('click', toggleMic);

function cleanupCall() {
  if (peerConnection) {
    peerConnection.close();
    peerConnection = null;
  }
  if (remoteVideo.srcObject) {
    remoteVideo.srcObject.getTracks().forEach(track => track.stop());
    remoteVideo.srcObject = null;
  }
  updateConnectionState(false);
  activeRoomLabel.textContent = 'None';
  roomId = null;
  // stop any in-progress speech when the call ends
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  lastSpokenSign = null;
  // hide in-call controls
  if (callActionsPanel) callActionsPanel.classList.remove('show');
  if (callOverlayControls) callOverlayControls.classList.remove('show');
  // remove in-call marker so video panels hide
  document.body.classList.remove('in-call');
}

if (endCallBtn) endCallBtn.addEventListener('click', async () => {
  if (!roomId) return;
  socket.emit('end-call', { token: authToken, room_id: roomId });
  cleanupCall();
  showToast('Call ended.', 'success');
});

if (overlayEndCallBtn) overlayEndCallBtn.addEventListener('click', async () => {
  if (!roomId) return;
  socket.emit('end-call', { token: authToken, room_id: roomId });
  cleanupCall();
  showToast('Call ended.', 'success');
});

if (overlayInterpreterBtn) overlayInterpreterBtn.addEventListener('click', () => {
  converterEnabled = !converterEnabled;
  if (converterSwitch) converterSwitch.checked = converterEnabled;
  updateConverterLabel();
  showToast(`Live converter ${converterEnabled ? 'enabled' : 'disabled'}`, 'info');
});

if (overlayFullscreenBtn) overlayFullscreenBtn.addEventListener('click', () => {
  try {
    const container = remoteVideo.parentElement;
    if (!document.fullscreenElement) container.requestFullscreen?.();
    else document.exitFullscreen?.();
  } catch (e) { console.error('Fullscreen error', e); }
});
// keep local preview visible when remote is fullscreen
document.addEventListener('fullscreenchange', () => {
  if (document.fullscreenElement) document.body.classList.add('in-fullscreen');
  else document.body.classList.remove('in-fullscreen');
});

function createPeerConnection() {
  const config = {
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
  };
  peerConnection = new RTCPeerConnection(config);

  peerConnection.ontrack = event => {
    remoteVideo.srcObject = event.streams[0];
    remoteOverlay.style.display = 'none';
  };

  peerConnection.onicecandidate = event => {
    if (event.candidate && roomId) {
      socket.emit('ice-candidate', { room_id: roomId, candidate: event.candidate });
    }
  };

  peerConnection.ondatachannel = event => {
    dataChannel = event.channel;
    dataChannel.onmessage = event => {
      try {
        const payload = JSON.parse(event.data);
        if (payload && payload.type === 'interpret') {
          const el = document.getElementById('interpretationOverlay');
          if (el) {
            el.textContent = `${payload.sign} (${Math.round(payload.confidence*100)}%)`;
            el.style.display = 'block';
            setTimeout(() => { el.style.display = 'none'; }, 3500);
          }
        }
      } catch (e) { console.log('dataChannel message', event.data); }
    };
    dataChannel.onopen = () => console.log('Data channel open.');
  };

  if (localStream) {
    localStream.getTracks().forEach(track => peerConnection.addTrack(track, localStream));
  }

  if (!dataChannel) {
    dataChannel = peerConnection.createDataChannel('call-data');
    dataChannel.onmessage = event => {
      try {
        const payload = JSON.parse(event.data);
        if (payload && payload.type === 'interpret') {
          const el = document.getElementById('interpretationOverlay');
          if (el) {
            el.textContent = `${payload.sign} (${Math.round(payload.confidence*100)}%)`;
            el.style.display = 'block';
            setTimeout(() => { el.style.display = 'none'; }, 3500);
          }
        }
      } catch (e) { console.log('dataChannel message', event.data); }
    };
    dataChannel.onopen = () => console.log('Data channel open.');
  }
}

async function joinRoom(code, createOnly = false) {
  const stream = await startLocalMedia();
  if (!stream) return;
  roomId = code;
  isCaller = createOnly;
  activeRoomLabel.textContent = roomId;
  updateConverterLabel();
  if (callOverlayControls) callOverlayControls.classList.add('show');

  remoteOverlay.style.display = 'flex';

  createPeerConnection();
  const token = authToken;

  socket.emit('join-room', {
    token,
    room_id: roomId,
    participant_type: createOnly ? 'host' : 'guest',
    interpreter_mode: converterEnabled
  });

  updateConnectionState(true);
  showToast(`Joined room ${roomId}`, 'success');
  // reveal in-call controls
  if (callActionsPanel) callActionsPanel.classList.add('show');
  if (callOverlayControls) callOverlayControls.classList.add('show');
  // mark body as in-call so video panels are visible
  document.body.classList.add('in-call');
}

if (startRoomBtn) {
  startRoomBtn.addEventListener('click', () => {
    console.log('Start Call clicked');
    const code = generateRoomCode();
    if (roomIdInput) roomIdInput.value = code;
    showToast(`Room code generated: ${code}`, 'success');
    try { joinRoom(code, true); } catch (e) { console.error('joinRoom error', e); showToast('Error starting call', 'error'); }
  });
}

if (joinRoomBtn && roomIdInput) {
  joinRoomBtn.addEventListener('click', () => {
    console.log('Join Call clicked');
    const code = roomIdInput.value.trim();
    if (!code) {
      showToast('Please enter a room code first.', 'error');
      return;
    }
    try { joinRoom(code); } catch (e) { console.error('joinRoom error', e); showToast('Error joining call', 'error'); }
  });
}

socket.on('connect', () => {
  showToast('Connected to signaling server.', 'success');
});

socket.on('connect_error', (err) => {
  console.error('Socket connect error', err);
  showToast('Signaling connection failed.', 'error');
});

socket.on('error', (err) => {
  console.error('Socket error', err);
  showToast('Signaling error', 'error');
});

function waitForSocketConnected(timeout = 5000) {
  return new Promise((resolve, reject) => {
    if (socket.connected) return resolve(true);
    const onConnect = () => { cleanup(); resolve(true); };
    const onError = (e) => { cleanup(); reject(e || new Error('connect_error')); };
    const timer = setTimeout(() => { cleanup(); reject(new Error('timeout')); }, timeout);
    function cleanup() { clearTimeout(timer); socket.off('connect', onConnect); socket.off('connect_error', onError); }
    socket.once('connect', onConnect);
    socket.once('connect_error', onError);
  });
}

socket.on('room-joined', async data => {
  if (!roomId || data.room_id !== roomId) return;
  showToast('Room participants updated.', 'info');
  const roomOwner = data.room_owner;
  const username = localStorage.getItem('username');
  const shouldOffer = roomOwner === username;
  console.log('room-joined', data, 'shouldOffer=', shouldOffer);
  if (data.participants.length >= 2 && shouldOffer && peerConnection && peerConnection.signalingState === 'stable') {
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    socket.emit('offer', {
      room_id: roomId,
      sdp: offer,
      target: 'peer'
    });
  }
});

socket.on('room-error', data => {
  console.warn('room-error', data);
  showToast(data.message || 'Room error from server', 'error');
});

socket.on('offer', async data => {
  if (!roomId || data.room_id !== roomId) return;
  if (!peerConnection) createPeerConnection();
  await peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp));
  const answer = await peerConnection.createAnswer();
  await peerConnection.setLocalDescription(answer);
  socket.emit('answer', {
    room_id: roomId,
    sdp: answer,
    target: 'peer'
  });
});

socket.on('answer', async data => {
  if (!roomId || data.room_id !== roomId) return;
  if (!peerConnection) return;
  await peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp));
});

socket.on('camera-changed', data => {
  if (!roomId || data.room_id !== roomId) return;
  const enabled = !!data.enabled;
  // When remote camera is turned off, show their overlay placeholder
  if (remoteOverlay) {
    remoteOverlay.style.display = enabled ? 'none' : 'flex';
  }
});

socket.on('mic-changed', data => {
  if (!roomId || data.room_id !== roomId) return;
  const enabled = !!data.enabled;
  // indicate remote mic state via a small badge in the overlay if present
  if (remoteOverlay) {
    const micBadge = document.getElementById('remoteMicBadge');
    if (micBadge) micBadge.textContent = enabled ? '' : '🔇';
  }
});

socket.on('ice-candidate', async data => {
  if (!peerConnection || !data.candidate) return;
  try {
    await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
  } catch (err) {
    console.error('Ice candidate error', err);
  }
});


socket.on('call-ended', data => {
  if (roomId && data.room_id === roomId) {
    cleanupCall();
    showToast('Call ended by remote participant.', 'info');
  }
});

socket.on('disconnect', () => {
  updateConnectionState(false);
  showToast('Signaling connection lost.', 'error');
});

const hands = new Hands({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.75, minTrackingConfidence: 0.65 });

const canvasCtx = localOverlay.getContext('2d');

function drawLocalLandmarks(results) {
  const width = localOverlay.width;
  const height = localOverlay.height;
  canvasCtx.clearRect(0, 0, width, height);
  if (!results.multiHandLandmarks) return;
  for (let i = 0; i < results.multiHandLandmarks.length; i += 1) {
    const landmarks = results.multiHandLandmarks[i];
    canvasCtx.strokeStyle = '#94a3b8';
    canvasCtx.lineWidth = 2;
    for (const [start, end] of [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]]) {
      const s = landmarks[start];
      const e = landmarks[end];
      canvasCtx.beginPath();
      canvasCtx.moveTo(s.x * width, s.y * height);
      canvasCtx.lineTo(e.x * width, e.y * height);
      canvasCtx.stroke();
    }
    for (const pt of landmarks) {
      canvasCtx.beginPath();
      canvasCtx.arc(pt.x * width, pt.y * height, 5, 0, Math.PI * 2);
      canvasCtx.fillStyle = 'rgba(56, 189, 248, 0.9)';
      canvasCtx.fill();
    }
  }
}

let localCamera = null;

async function startDetection() {
  if (!localStream) return;
  const videoTrack = localStream.getVideoTracks()[0];
  if (!videoTrack) return;

  localVideo.srcObject = localStream;
  await localVideo.play();

  localVideo.style.display = 'block';
  localOverlay.style.display = 'block';

  localOverlay.width = localVideo.videoWidth || 1280;
  localOverlay.height = localVideo.videoHeight || 720;

  localCamera = new Camera(localVideo, {
    onFrame: async () => {
      if (!isCameraOn) return;
      await hands.send({ image: localVideo });
    },
    width: 1280,
    height: 720
  });
  await localCamera.start();
}

hands.onResults(results => {
  if (!localOverlay.width || !localOverlay.height) {
    localOverlay.width = localVideo.videoWidth || 1280;
    localOverlay.height = localVideo.videoHeight || 720;
  }
  canvasCtx.clearRect(0, 0, localOverlay.width, localOverlay.height);
  if (results.image) {
    canvasCtx.drawImage(results.image, 0, 0, localOverlay.width, localOverlay.height);
  }
  drawLocalLandmarks(results);
  if (converterEnabled && Date.now() - lastPredictionTime > MIN_PREDICT_INTERVAL) {
    lastPredictionTime = Date.now();
    predictSign(results);
  }
});

function extractFeatures(results) {
  const left = new Array(63).fill(0.0);
  const right = new Array(63).fill(0.0);
  if (!results.multiHandLandmarks || !results.multiHandedness) return left.concat(right);
  for (let i = 0; i < results.multiHandLandmarks.length; i += 1) {
    const handLandmarks = results.multiHandLandmarks[i];
    const label = results.multiHandedness[i].label || '';
    const coords = [];
    for (const lm of handLandmarks) coords.push(lm.x, lm.y, lm.z);
    if (label === 'Left') left.splice(0, coords.length, ...coords);
    if (label === 'Right') right.splice(0, coords.length, ...coords);
  }
  return left.concat(right);
}

async function predictSign(results) {
  if (!results || !converterEnabled) return;

  // Real check: were any hands actually detected this frame?
  const hasHands = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
  if (!hasHands) {
    signLabelValue.textContent = 'No hands detected';
    confidenceValue.textContent = '0%';
    frameRateValue.textContent = '—';
    lastSpokenSign = null;  // reset so the next real sign always gets spoken fresh
    return;
  }

  const features = extractFeatures(results);
  if (!features || features.length !== 126) {
    signLabelValue.textContent = 'No hands detected';
    confidenceValue.textContent = '0%';
    frameRateValue.textContent = '—';
    return;
  }
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': authToken },
      body: JSON.stringify({ features })
    });
    const data = await response.json();
    if (response.ok && data.success) {
      signLabelValue.textContent = data.predicted_class;
      confidenceValue.textContent = `${Math.round(data.confidence * 100)}%`;
      frameRateValue.textContent = `${Math.round(1000 / Math.max(1, Date.now() - lastPredictionTime))} fps`;

      // ---- Speak the predicted sign aloud (with cooldown / confidence gate) ----
      const now = Date.now();
      const changedSign = data.predicted_class !== lastSpokenSign;
      const cooldownPassed = now - lastSpokenTime > SPEAK_COOLDOWN_MS;
      if (data.confidence >= MIN_CONFIDENCE_TO_SPEAK && (changedSign || cooldownPassed)) {
        speakSign(data.predicted_class);
        lastSpokenSign = data.predicted_class;
        lastSpokenTime = now;
      }

      try {
        if (dataChannel && dataChannel.readyState === 'open') {
          dataChannel.send(JSON.stringify({ type: 'interpret', sign: data.predicted_class, confidence: data.confidence }));
        }
      } catch (e) { console.error('send interpret', e); }
    } else {
      const message = data?.message || 'Prediction failed';
      console.warn('Predict error:', message);
      signLabelValue.textContent = 'Recognition failed';
      confidenceValue.textContent = '0%';
      frameRateValue.textContent = '—';
    }
  } catch (err) {
    console.error('Prediction error', err);
    const errorMessage = err?.message || 'Prediction network error';
    signLabelValue.textContent = 'Recognition failed';
    confidenceValue.textContent = '0%';
    frameRateValue.textContent = '—';
    showToast(errorMessage, 'error');
  }
}

async function initApp() {
  updateConverterLabel();
  updateAudioLabel();
  const valid = await verifyProfile();
  if (!valid) return;
  await fetchCallHistory();
  await startLocalMedia();
  await startDetection();
}

refreshHistoryBtn.addEventListener('click', fetchCallHistory);

initApp();