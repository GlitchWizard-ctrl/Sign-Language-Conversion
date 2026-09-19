// Live SignCast client script
const authToken = localStorage.getItem('authToken');
if (!authToken) {
  window.location.href = 'login.html';
}

const socket = io(window.location.origin, {
  autoConnect: false,
  transports: ['websocket'],
  upgrade: false,
  auth: { token: authToken }
});
let localStream = null;
let peerConnections = {};
let dataChannels = {};
let peerUsernames = {};
let roomId = null;
let isCaller = false;
let isCameraOn = false;
let isMicOn = false;
let converterEnabled = true;
let lastPredictionTime = 0;
let lastPredictionFailureAt = 0;
const MIN_PREDICT_INTERVAL = 1000;
const PREDICT_FAILURE_BACKOFF_MS = 4000;

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
const localOverlay = document.getElementById('localOverlay');
const remotesContainer = document.getElementById('remotesContainer');
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
const overlayToggleCamLocal = document.getElementById('overlayToggleCamLocal');
const overlayToggleMicLocal = document.getElementById('overlayToggleMicLocal');
const overlayEndCallBtnLocal = document.getElementById('overlayEndCallBtnLocal');
const overlayInterpreterBtnLocal = document.getElementById('overlayInterpreterBtnLocal');
const overlayFullscreenBtnLocal = document.getElementById('overlayFullscreenBtnLocal');
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
// user dropdown elements
const userPillBtn = document.getElementById('userPillBtn');
const userMenu = document.getElementById('userMenu');

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

usernameLabel.textContent = localStorage.getItem('fullname') || localStorage.getItem('username') || 'Guest';
userRoleLabel.textContent = (localStorage.getItem('role') || 'user').toUpperCase();
adminBanner.style.display = 'none';

function buildAuthHeaders(extra = {}) {
  const token = localStorage.getItem('authToken');
  return {
    ...extra,
    'Authorization': token ? `Bearer ${token}` : ''
  };
}

signOutBtn.addEventListener('click', async () => {
  await fetch('/api/logout', {
    method: 'POST',
    headers: buildAuthHeaders()
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
      headers: buildAuthHeaders()
    });
    const data = await response.json();
    if (!response.ok || !data.success) {
      throw new Error('Unauthorized');
    }

    // Admins get their own dashboard — bail out before any camera/socket
    // setup runs on this page.
    if (data.role === 'admin') {
      localStorage.setItem('role', 'admin');
      window.location.href = 'admin.html';
      return false;
    }

    usernameLabel.textContent = data.fullname || data.username || 'Guest';
    userRoleLabel.textContent = (data.role || 'user').toUpperCase();
    adminBanner.style.display = 'none';
    localStorage.setItem('role', 'user');
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

function updateCameraUi() {
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
}

async function stopCamera() {
  if (!localStream) return;
  localStream.getVideoTracks().forEach(track => {
    track.stop(); // Releases the physical webcam indicator and device.
    localStream.removeTrack(track);
  });
  if (localCamera && localCamera.stop) {
    try { await localCamera.stop(); } catch (e) { /* ignore */ }
    localCamera = null;
  }
  isCameraOn = false;
  updateCameraUi();
  if (socket && socket.connected && roomId) {
    socket.emit('camera-toggle', { room_id: roomId, enabled: false });
  }
}

async function startCamera() {
  if (!roomId) return;
  try {
    const cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    const videoTrack = cameraStream.getVideoTracks()[0];
    if (!videoTrack) return;
    // ensure we have a localStream object
    if (!localStream) localStream = new MediaStream();
    localStream.addTrack(videoTrack);
    localVideo.srcObject = localStream;

    // replace/add video sender for each peer connection
    for (const sid of Object.keys(peerConnections)) {
      const pc = peerConnections[sid];
      try {
        const sender = pc.getSenders().find(item => item.track?.kind === 'video');
        if (sender) await sender.replaceTrack(videoTrack);
        else pc.addTrack(videoTrack, localStream);
      } catch (e) { console.warn('replace/add track failed for', sid, e); }
    }

    isCameraOn = true;
    updateCameraUi();
    if (socket?.connected) socket.emit('camera-toggle', { room_id: roomId, enabled: true });
  } catch (err) {
    console.error('Camera start failed', err);
    showToast('Unable to turn on the camera.', 'error');
  }
}

async function toggleCamera() {
  if (!roomId) return;
  if (isCameraOn) await stopCamera();
  else await startCamera();
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
  // close all peer connections and remove remote videos
  for (const sid of Object.keys(peerConnections)) {
    try { peerConnections[sid].close(); } catch (e) {}
    delete peerConnections[sid];
    delete dataChannels[sid];
    removeRemoteVideo(sid);
  }
  if (localCamera?.stop) localCamera.stop();
  localCamera = null;
  if (localStream) {
    localStream.getTracks().forEach(track => track.stop());
    localStream = null;
  }
  // reset caption stream
  const capEl = document.getElementById('captionHistory');
  if (capEl) capEl.innerHTML = '';
  lastCaptionAuthor = null;
  localVideo.srcObject = null;
  isCameraOn = false;
  isMicOn = false;
  updateCameraUi();
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

function leaveCurrentCall({ endedForEveryone = false } = {}) {
  if (!roomId) return;
  const payload = { token: authToken, room_id: roomId, room: roomId };
  if (endedForEveryone) socket.emit('end-call', payload);
  else socket.emit('leave-call', payload);
  cleanupCall();
}

if (endCallBtn) endCallBtn.addEventListener('click', async () => {
  if (!roomId) return;
  leaveCurrentCall({ endedForEveryone: true });
  showToast('Call ended.', 'success');
});

if (overlayEndCallBtn) overlayEndCallBtn.addEventListener('click', async () => {
  if (!roomId) return;
  leaveCurrentCall({ endedForEveryone: true });
  showToast('Call ended.', 'success');
});

if (overlayEndCallBtnLocal) overlayEndCallBtnLocal.addEventListener('click', async () => {
  if (!roomId) return;
  leaveCurrentCall({ endedForEveryone: true });
  showToast('Call ended.', 'success');
});

window.addEventListener('beforeunload', () => {
  if (roomId && socket?.connected) {
    socket.emit('leave-call', { token: authToken, room_id: roomId, room: roomId });
  }
});

if (overlayToggleCamLocal) overlayToggleCamLocal.addEventListener('click', toggleCamera);
if (overlayToggleMicLocal) overlayToggleMicLocal.addEventListener('click', toggleMic);
if (overlayInterpreterBtnLocal) overlayInterpreterBtnLocal.addEventListener('click', () => {
  converterEnabled = !converterEnabled;
  if (converterSwitch) converterSwitch.checked = converterEnabled;
  updateConverterLabel();
  showToast(`Live converter ${converterEnabled ? 'enabled' : 'disabled'}`, 'info');
});
if (overlayFullscreenBtnLocal) overlayFullscreenBtnLocal.addEventListener('click', () => {
  try {
    const localCard = document.getElementById('localCameraCard') || document.querySelector('.app-container') || document.body;
    if (!document.fullscreenElement) localCard.requestFullscreen?.();
    else document.exitFullscreen?.();
  } catch (e) { console.error('Fullscreen error', e); }
});

if (overlayInterpreterBtn) overlayInterpreterBtn.addEventListener('click', () => {
  converterEnabled = !converterEnabled;
  if (converterSwitch) converterSwitch.checked = converterEnabled;
  updateConverterLabel();
  showToast(`Live converter ${converterEnabled ? 'enabled' : 'disabled'}`, 'info');
});

if (overlayFullscreenBtn) overlayFullscreenBtn.addEventListener('click', () => {
  try {
    const remoteContainer = document.getElementById('remotesContainer') || document.querySelector('.app-container') || document.body;
    if (!document.fullscreenElement) remoteContainer.requestFullscreen?.();
    else document.exitFullscreen?.();
  } catch (e) { console.error('Fullscreen error', e); }
});
// keep local preview visible when remote is fullscreen
document.addEventListener('fullscreenchange', () => {
  if (document.fullscreenElement) document.body.classList.add('in-fullscreen');
  else document.body.classList.remove('in-fullscreen');
});

function createRemoteVideoElement(sid, username) {
  const card = document.createElement('div');
  card.className = 'remote-card';
  card.id = `remote-${sid}`;
  card.setAttribute('data-sid', sid);
  const video = document.createElement('video');
  video.autoplay = true;
  video.playsinline = true;
  card.appendChild(video);
  const meta = document.createElement('div'); meta.className = 'remote-meta'; meta.textContent = username || sid;
  card.appendChild(meta);
  remotesContainer.appendChild(card);
  console.log('[client] createRemoteVideoElement', sid, username);
  updateRemotePlaceholder();
  return video;
}

function removeRemoteVideo(sid) {
  const el = document.getElementById(`remote-${sid}`);
  if (el) el.remove();
  updateRemotePlaceholder();
}

function updateRemotePlaceholder() {
  const container = document.getElementById('remotesContainer');
  const placeholder = document.getElementById('remoteOverlay');
  if (!container || !placeholder) return;
  const remotes = container.querySelectorAll('.remote-card');
  if (remotes.length === 0) {
    placeholder.style.display = 'flex';
    updateVideoLayout(true);
  } else {
    placeholder.style.display = 'none';
    updateVideoLayout(false);
  }
}

// Fallback UX: if no remotes appear within X seconds after joining, show actionable message
let _remoteWaitTimer = null;
const REMOTE_WAIT_MS = 8000;
function startRemoteWaitTimer() {
  clearRemoteWaitTimer();
  const placeholder = document.getElementById('remoteOverlay');
  if (!placeholder) return;
  placeholder.textContent = 'Waiting for participant...';
  _remoteWaitTimer = setTimeout(() => {
    // if still no remotes, show share/copy action
    const container = document.getElementById('remotesContainer');
    if (!container) return;
    const remotes = container.querySelectorAll('.remote-card');
    if (remotes.length > 0) return;
    placeholder.textContent = 'No other participants found.';
    // add copy-code button
    let btn = document.getElementById('copyRoomCodeBtn');
    if (!btn) {
      btn = document.createElement('button');
      btn.id = 'copyRoomCodeBtn';
      btn.className = 'btn-small';
      btn.textContent = 'Copy room code';
      btn.style.marginTop = '12px';
      btn.onclick = async () => {
        try {
          await navigator.clipboard.writeText(roomId || '');
          showToast('Room code copied to clipboard', 'success');
        } catch (e) { showToast('Unable to copy', 'error'); }
      };
      placeholder.appendChild(btn);
    }
  }, REMOTE_WAIT_MS);
}

function clearRemoteWaitTimer() {
  if (_remoteWaitTimer) { clearTimeout(_remoteWaitTimer); _remoteWaitTimer = null; }
  const placeholder = document.getElementById('remoteOverlay');
  if (!placeholder) return;
  // remove copy button if present
  const btn = document.getElementById('copyRoomCodeBtn');
  if (btn) btn.remove();
  // restore default text if empty
  if (!placeholder.textContent || placeholder.textContent.trim() === '') placeholder.textContent = 'Waiting for participant...';
}

function updateVideoLayout(singleLocal) {
  if (singleLocal) document.body.classList.add('single-local');
  else document.body.classList.remove('single-local');
}

// Support older server event name 'existing-peers' used by some join flows
socket.on('existing-peers', async data => {
  if (!roomId) return;
  const peers = data.peers || [];
  for (const p of peers) {
    if (!p || !p.sid) continue;
    const sid = p.sid;
    if (sid === socket.id) continue;
    if (peerConnections[sid]) continue;
    const pc = createPeerConnection(sid, p.username, true);
    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      socket.emit('offer', { room_id: roomId, sdp: offer, to: sid });
    } catch (e) { console.error('Failed to create/send offer to', sid, e); }
  }
});

function setupDataChannel(sid, ch) {
  ch.onopen = () => { console.log('Data channel open', sid); };
  ch.onclose = () => { console.log('Data channel closed', sid); };
  ch.onerror = (e) => { console.error('Data channel error', sid, e); };
  ch.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg && msg.type === 'interpret') {
        // show live interpretation in caption panel immediately
        const who = peerUsernames[sid] || 'remote';
        appendCaption(who, msg.sign, msg.confidence, null);
      }
    } catch (e) { console.error('data channel msg', e); }
  };
}

function createPeerConnection(sid, username, initiator = false) {
  if (peerConnections[sid]) return peerConnections[sid];
  const pc = new RTCPeerConnection();
  peerConnections[sid] = pc;
  peerUsernames[sid] = username || sid;

  // add local tracks if available
  if (localStream) {
    for (const track of localStream.getTracks()) pc.addTrack(track, localStream);
  }

  pc.onicecandidate = (e) => {
    if (e.candidate) {
      socket.emit('ice-candidate', { to: sid, room_id: roomId, candidate: e.candidate });
    }
  };

  pc.ontrack = (e) => {
    let video = document.getElementById(`remoteVideo-${sid}`);
    if (!video) {
      video = createRemoteVideoElement(sid, username);
      video.id = `remoteVideo-${sid}`;
    }
    try {
      console.log('[client] ontrack for', sid, 'streams:', e.streams && e.streams.length);
      video.srcObject = e.streams[0];
      video.play().catch(() => {});
    } catch (err) { console.error('set remote stream', err); }
  };

  if (initiator) {
    try {
      const dc = pc.createDataChannel('sign-data');
      setupDataChannel(sid, dc);
      dataChannels[sid] = dc;
    } catch (e) { console.warn('createDataChannel failed', e); }
  } else {
    pc.ondatachannel = (ev) => {
      const dc = ev.channel;
      setupDataChannel(sid, dc);
      dataChannels[sid] = dc;
    };
  }

  return pc;
}

async function joinRoom(code, createOnly = false) {
  const stream = await startLocalMedia();
  if (!stream) return;
  roomId = code;
  isCaller = createOnly;
  activeRoomLabel.textContent = roomId;
  updateConverterLabel();
  if (callOverlayControls) callOverlayControls.classList.add('show');

  await startDetection();
  const token = authToken;

  socket.emit('join-room', {
    token,
    room_id: roomId,
    participant_type: createOnly ? 'host' : 'guest',
    interpreter_mode: converterEnabled
  });

  // Start fallback timer to show share/copy action if no remotes connect
  startRemoteWaitTimer();

  // Ensure layout reflects current remote count immediately
  updateRemotePlaceholder();

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
  console.log('[client] room-joined', data);
  // Create peer connections and send offers to each existing participant
  const peers = data.peers || [];
  // peers updated — clear fallback timer (we have peer info)
  clearRemoteWaitTimer();
  for (const p of peers) {
    if (!p || !p.sid) continue;
    const sid = p.sid;
    if (sid === socket.id) continue;
    if (peerConnections[sid]) continue;
    const pc = createPeerConnection(sid, p.username, true);
    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      socket.emit('offer', { room_id: roomId, sdp: offer, to: sid });
    } catch (e) { console.error('Failed to create/send offer to', sid, e); }
  }
});

socket.on('peer-joined', data => {
  if (!data || !data.sid) return;
  if (roomId && data.room_id && data.room_id !== roomId) return;
  console.log('[client] peer-joined', data);
  clearRemoteWaitTimer();
  if (!document.getElementById(`remote-${data.sid}`)) {
    createRemoteVideoElement(data.sid, data.username || data.sid);
  }
  updateRemotePlaceholder();
});

socket.on('room-error', data => {
  console.warn('room-error', data);
  showToast(data.message || 'Room error from server', 'error');
});

socket.on('offer', async data => {
  if (!roomId || data.room_id !== roomId) return;
  const from = data.from;
  if (!from) return;
  const username = data.username || null;
  const pc = createPeerConnection(from, username, false);
  try {
    await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    socket.emit('answer', { room_id: roomId, sdp: answer, to: from });
  } catch (e) { console.error('Error handling offer', e); }
});

socket.on('answer', async data => {
  if (!roomId || data.room_id !== roomId) return;
  const from = data.from;
  if (!from) return;
  const pc = peerConnections[from];
  if (!pc) return console.warn('No peerConnection for answer from', from);
  try { await pc.setRemoteDescription(new RTCSessionDescription(data.sdp)); } catch (e) { console.error(e); }
});

socket.on('camera-changed', data => {
  if (!roomId || data.room_id !== roomId) return;
  const enabled = !!data.enabled;
  const sid = data.sid;
  if (!sid) return;
  const card = document.getElementById(`remote-${sid}`);
  if (card) {
    if (!enabled) {
      // show a simple overlay by reducing opacity
      card.style.opacity = '0.25';
    } else {
      card.style.opacity = '1';
    }
  }
});

socket.on('mic-changed', data => {
  if (!roomId || data.room_id !== roomId) return;
  const enabled = !!data.enabled;
  const sid = data.sid;
  if (!sid) return;
  const card = document.getElementById(`remote-${sid}`);
  if (card) {
    let badge = card.querySelector('.remote-mic-badge');
    if (!badge) {
      badge = document.createElement('div'); badge.className = 'remote-mic-badge'; badge.style.position='absolute'; badge.style.right='8px'; badge.style.top='8px'; badge.style.fontSize='14px'; card.appendChild(badge);
    }
    badge.textContent = enabled ? '' : '🔇';
  }
});

socket.on('ice-candidate', async data => {
  const from = data.from;
  if (!from || !data.candidate) return;
  const pc = peerConnections[from];
  if (!pc) return;
  try { await pc.addIceCandidate(new RTCIceCandidate(data.candidate)); } catch (err) { console.error('Ice candidate error', err); }
});

socket.on('peer-left', data => {
  if (!data) return;
  if (roomId && data.room_id && data.room_id !== roomId) return;
  const sid = data.sid;
  if (!sid) return;
  if (peerConnections[sid]) {
    try { peerConnections[sid].close(); } catch (e) {}
    delete peerConnections[sid];
  }
  removeRemoteVideo(sid);
  if (data.username) {
    showToast(`${data.username} left the call.`, 'info');
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

// Receive caption history for the room upon joining
socket.on('caption-history', data => {
  const capEl = document.getElementById('captionHistory');
  if (!capEl || !data || !data.captions) return;
  capEl.innerHTML = '';
  for (const c of data.captions) {
    appendCaption(c.username, c.text, c.confidence, c.created_at);
  }
});

socket.on('sign-caption', data => {
  if (!data) return;
  appendCaption(data.username, data.text, data.confidence, null);
});

function appendCaption(username, text, confidence, created_at) {
  const capEl = document.getElementById('captionHistory');
  if (!capEl) return;
  // Render captions as a continuous inline stream.
  const remoteCount = (document.getElementById('remotesContainer')?.querySelectorAll('.remote-card') || []).length;
  const onlyLocal = remoteCount === 0;
  let stream = document.getElementById('captionStream');
  if (!stream) {
    stream = document.createElement('div');
    stream.id = 'captionStream';
    stream.className = 'caption-stream';
    capEl.innerHTML = '';
    capEl.appendChild(stream);
    lastCaptionAuthor = null;
  }

  const safeText = text || '';
  const now = Date.now();
  // avoid repeating identical caption from same user within short window
  if (safeText && lastCaptionText === safeText && lastCaptionAuthor === username && (now - lastCaptionTime) < 3000) {
    return;
  }
  if (onlyLocal) {
    const span = document.createElement('span'); span.className = 'caption-word'; span.textContent = (stream.childElementCount ? ' ' : '') + safeText;
    stream.appendChild(span);
    lastCaptionText = safeText;
    lastCaptionTime = now;
  } else {
    const user = username || 'unknown';
    if (lastCaptionAuthor !== user) {
      // new speaker
      const sep = document.createElement('span'); sep.className = 'caption-sep'; sep.textContent = '\u00a0';
      const label = document.createElement('span'); label.className = 'caption-who'; label.textContent = user + ':';
      const word = document.createElement('span'); word.className = 'caption-word'; word.textContent = ' ' + safeText;
      stream.appendChild(sep);
      stream.appendChild(label);
      stream.appendChild(word);
      lastCaptionAuthor = user;
    } else {
      const word = document.createElement('span'); word.className = 'caption-word'; word.textContent = ' ' + safeText;
      stream.appendChild(word);
    }
      lastCaptionText = safeText;
      lastCaptionTime = now;
  }
  capEl.scrollTop = capEl.scrollHeight;
}

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
  if (localCamera) return;
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
  const now = Date.now();
  const failureBackoffActive = now - lastPredictionFailureAt < PREDICT_FAILURE_BACKOFF_MS;
  if (converterEnabled && !failureBackoffActive && now - lastPredictionTime > MIN_PREDICT_INTERVAL) {
    lastPredictionTime = now;
    predictSign(results);
  }
});

function extractFeatures(results) {
  const left = new Array(63).fill(0.0);
  const right = new Array(63).fill(0.0);
  if (!results || !results.multiHandLandmarks) return left.concat(right);

  const handedness = Array.isArray(results.multiHandedness) ? results.multiHandedness : [];
  for (let i = 0; i < results.multiHandLandmarks.length; i += 1) {
    const handLandmarks = results.multiHandLandmarks[i];
    const coords = [];
    for (const lm of handLandmarks) coords.push(lm.x, lm.y, lm.z);
    if (coords.length !== 63) continue;

    const labelObj = handedness[i];
    const rawLabel = (labelObj && labelObj.label) ? String(labelObj.label).trim().toLowerCase() : '';
    let handType = rawLabel;

    if (!handType) {
      const wristX = handLandmarks[0]?.x ?? 0.5;
      handType = wristX < 0.5 ? 'left' : 'right';
    }

    if (handType === 'left') left.splice(0, 63, ...coords);
    else if (handType === 'right') right.splice(0, 63, ...coords);
    else {
      const wristX = handLandmarks[0]?.x ?? 0.5;
      if (wristX < 0.5) left.splice(0, 63, ...coords);
      else right.splice(0, 63, ...coords);
    }
  }

  const normalized = left.concat(right);
  return normalized.some(v => Math.abs(v) > 1e-6) ? normalized : new Array(126).fill(0.0);
}

async function predictSign(results) {
  if (!results || !converterEnabled) return;

  const hasHands = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
  if (!hasHands) {
    signLabelValue.textContent = 'No hands detected';
    confidenceValue.textContent = '0%';
    frameRateValue.textContent = '—';
    lastSpokenSign = null;
    return;
  }

  const features = extractFeatures(results);
  if (!features || features.length !== 126 || features.every(v => Math.abs(v) < 1e-6)) {
    signLabelValue.textContent = 'No hands detected';
    confidenceValue.textContent = '0%';
    frameRateValue.textContent = '—';
    return;
  }

  try {
    console.log('[predict] features length', features.length, 'sample:', features.slice(0,10));
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ features })
    });

    if (!response.ok) {
      const text = await response.text().catch(() => '');
      let message = 'Prediction failed';
      try {
        const parsed = text ? JSON.parse(text) : null;
        if (parsed && parsed.message) message = parsed.message;
      } catch (e) {}

      if (response.status === 401 || response.status === 403 || response.status === 429) {
        lastPredictionFailureAt = Date.now();
      }

      console.warn('Predict error:', response.status, message);
      signLabelValue.textContent = message;
      confidenceValue.textContent = '0%';
      frameRateValue.textContent = '—';
      if (response.status !== 429) showToast(message, 'error');
      return;
    }

    const data = await response.json();
    if (data.success) {
      signLabelValue.textContent = data.predicted_class;
      confidenceValue.textContent = `${Math.round(data.confidence)}%`;
      const now = Date.now();
      frameRateValue.textContent = `${Math.round(1000 / Math.max(1, now - lastPredictionTime))} fps`;

      const changedSign = data.predicted_class !== lastSpokenSign;
      const cooldownPassed = now - lastSpokenTime > SPEAK_COOLDOWN_MS;
      if (data.confidence >= MIN_CONFIDENCE_TO_SPEAK * 100 && (changedSign || cooldownPassed)) {
        enqueueTeleprompter(data.predicted_class, localStorage.getItem('username') || 'You', data.confidence);
        lastSpokenSign = data.predicted_class;
        lastSpokenTime = now;
      }

      try {
        const msg = JSON.stringify({ type: 'interpret', sign: data.predicted_class, confidence: data.confidence });
        for (const sid in dataChannels) {
          const ch = dataChannels[sid];
          if (ch && ch.readyState === 'open') {
            ch.send(msg);
          }
        }
      } catch (e) { console.error('send interpret', e); }

      try {
        if (roomId) socket.emit('publish-caption', { room: roomId, text: data.predicted_class, confidence: data.confidence });
      } catch (e) { console.error('publish-caption', e); }
    } else {
      const message = data?.message || 'Prediction failed';
      console.warn('Predict error:', message);
      signLabelValue.textContent = message;
      confidenceValue.textContent = '0%';
      frameRateValue.textContent = '—';
      lastPredictionFailureAt = Date.now();
      showToast(message, 'error');
    }
  } catch (err) {
    console.error('Prediction error', err);
    const errorMessage = err?.message || 'Prediction network error';
    signLabelValue.textContent = 'Recognition failed';
    confidenceValue.textContent = '0%';
    frameRateValue.textContent = '—';
    lastPredictionFailureAt = Date.now();
    showToast(errorMessage, 'error');
  }
}

async function initApp() {
  updateConverterLabel();
  updateAudioLabel();
  const valid = await verifyProfile();
  if (!valid) return;
  try { socket.connect(); } catch (e) { console.warn('Socket connect failed', e); }
}

initApp();

// ------------------------- Teleprompter + speech queue -------------------------
const teleprompterEl = document.getElementById('teleprompter');
let teleQueue = [];
let teleSpeaking = false;
let lastCaptionAuthor = null;
let lastCaptionText = null;
let lastCaptionTime = 0;

function enqueueTeleprompter(text, who = 'You', confidence = 100) {
  if (!text) return;
  teleQueue.push({ text, who, confidence });
  if (!teleSpeaking) processTeleQueue();
}

function processTeleQueue() {
  if (teleQueue.length === 0) { teleSpeaking = false; teleprompterEl.innerHTML = '&nbsp;'; return; }
  teleSpeaking = true;
  const item = teleQueue.shift();
  speakAndDisplay(item.text, item.who).then(() => { teleSpeaking = false; processTeleQueue(); }).catch(() => { teleSpeaking = false; processTeleQueue(); });
}

function speakAndDisplay(text, who) {
  return new Promise((resolve) => {
    if (!teleprompterEl) { resolve(); return; }
    // Prepare utterance but do not display teleprompter text until speech actually starts
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.0; utter.pitch = 1.0;
    let revealed = 0;
    let fallbackTimer = null;

    const showPartial = (charIndex) => {
      if (!teleprompterEl) return;
      revealed = Math.max(revealed, charIndex || 0);
      const shown = text.slice(0, revealed);
      const pending = text.slice(revealed);
      teleprompterEl.innerHTML = `<span class="revealed">${escapeHtml(shown)}</span><span class="pending">${escapeHtml(pending)}</span>`;
    };

    // onboundary progressive reveal
    utter.onboundary = (ev) => {
      try { showPartial(ev.charIndex + (ev.charLength || 0)); } catch (e) {}
    };

    // When speech actually starts, initialize display and, if onboundary unsupported, start fallback reveal
    utter.onstart = () => {
      // cancel any fallback guard
      if (fallbackTimer) { clearTimeout(fallbackTimer); fallbackTimer = null; }
      teleprompterEl.textContent = '';
      // If onboundary unsupported, run a timed reveal based on estimated duration
      if (!('onboundary' in SpeechSynthesisUtterance.prototype)) {
        const words = text.split(/\s+/).filter(Boolean);
        let idx = 0;
        const total = Math.max(1, words.length);
        const estDuration = Math.max(600, words.length * 350);
        const step = Math.max(1, Math.floor(text.length / total));
        const interval = Math.max(50, Math.floor(estDuration / total));
        const timer = setInterval(() => {
          idx += 1;
          const chars = Math.min(text.length, idx * step);
          showPartial(chars);
          if (chars >= text.length) { clearInterval(timer); }
        }, interval);
      }
    };

    utter.onend = () => { showPartial(text.length); setTimeout(resolve, 150); };

    // If speech never starts due to autoplay/policy, cancel display: wait a short guard and then resolve without showing
    fallbackTimer = setTimeout(() => {
      // speech did not start — do not show teleprompter
      resolve();
    }, 800);

    // speak
    try {
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utter);
    } catch (e) {
      console.error('Speech speak failed', e);
      if (fallbackTimer) { clearTimeout(fallbackTimer); }
      resolve();
    }
  });
}

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
