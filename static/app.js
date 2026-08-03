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
let interpreterEnabled = false;
let lastPredictionTime = 0;
const MIN_PREDICT_INTERVAL = 200;

const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const localOverlay = document.getElementById('localOverlay');
const remoteOverlay = document.getElementById('remoteOverlay');
const connectionStatus = document.getElementById('connectionStatus');
const activeRoomLabel = document.getElementById('activeRoomLabel');
const interpreterLabel = document.getElementById('interpreterLabel');
const signLabelValue = document.getElementById('signLabelValue');
const confidenceValue = document.getElementById('confidenceValue');
const frameRateValue = document.getElementById('frameRateValue');

const roomIdInput = document.getElementById('roomIdInput');
const generateRoomBtn = document.getElementById('generateRoomBtn');
const startRoomBtn = document.getElementById('startRoomBtn');
const joinRoomBtn = document.getElementById('joinRoomBtn');
const toggleCamBtn = document.getElementById('toggleCamBtn');
const toggleMicBtn = document.getElementById('toggleMicBtn');
const endCallBtn = document.getElementById('endCallBtn');
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');
const interpreterSwitch = document.getElementById('interpreterSwitch');
const converterSwitch = document.getElementById('converterSwitch');

const toastContainer = document.getElementById('toastContainer');
const usernameLabel = document.getElementById('usernameLabel');
const userRoleLabel = document.getElementById('userRoleLabel');
const adminBanner = document.getElementById('adminBanner');
const signOutBtn = document.getElementById('signOutBtn');
const historyBody = document.getElementById('historyBody');

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

function updateInterpreterLabel() {
  interpreterLabel.textContent = interpreterEnabled ? 'On' : 'Off';
}

function generateRoomCode() {
  return Math.random().toString(36).substring(2, 8).toUpperCase();
}

generateRoomBtn.addEventListener('click', () => {
  roomIdInput.value = generateRoomCode();
  showToast('Room code generated', 'success');
});

interpreterSwitch.addEventListener('change', () => {
  interpreterEnabled = interpreterSwitch.checked;
  updateInterpreterLabel();
  if (peerConnection && roomId) {
    socket.emit('interpreter-toggle', { room_id: roomId, enabled: interpreterEnabled });
  }
});

converterSwitch.addEventListener('change', () => {
  converterEnabled = converterSwitch.checked;
  showToast(converterEnabled ? 'Live converter enabled' : 'Live converter disabled', 'info');
});

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
    toggleCamBtn.textContent = 'Camera OFF';
    toggleMicBtn.textContent = 'Mic OFF';
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
  toggleCamBtn.textContent = isCameraOn ? 'Camera OFF' : 'Camera ON';
  localVideo.classList.toggle('video-muted', !isCameraOn);
  document.getElementById('localCameraHint').style.display = isCameraOn ? 'none' : 'flex';
}

function toggleMic() {
  if (!localStream) return;
  const audioTrack = localStream.getAudioTracks()[0];
  if (!audioTrack) return;
  audioTrack.enabled = !audioTrack.enabled;
  isMicOn = audioTrack.enabled;
  toggleMicBtn.textContent = isMicOn ? 'Mic OFF' : 'Mic ON';
}

toggleCamBtn.addEventListener('click', toggleCamera);
toggleMicBtn.addEventListener('click', toggleMic);

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
}

endCallBtn.addEventListener('click', async () => {
  if (!roomId) return;
  socket.emit('end-call', { token: authToken, room_id: roomId });
  cleanupCall();
  showToast('Call ended.', 'success');
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
    dataChannel.onmessage = event => console.log('Data channel message:', event.data);
    dataChannel.onopen = () => console.log('Data channel open.');
  };

  if (localStream) {
    localStream.getTracks().forEach(track => peerConnection.addTrack(track, localStream));
  }

  if (!dataChannel) {
    dataChannel = peerConnection.createDataChannel('call-data');
    dataChannel.onmessage = event => console.log('Data channel message:', event.data);
    dataChannel.onopen = () => console.log('Data channel open.');
  }
}

async function joinRoom(code, createOnly = false) {
  const stream = await startLocalMedia();
  if (!stream) return;

  roomId = code;
  isCaller = createOnly;
  activeRoomLabel.textContent = roomId;
  interpreterLabel.textContent = interpreterEnabled ? 'On' : 'Off';

  if (!socket.connected) {
    socket.connect();
  }

  createPeerConnection();
  const token = authToken;

  socket.emit('join-room', {
    token,
    room_id: roomId,
    participant_type: createOnly ? 'host' : 'guest',
    interpreter_mode: interpreterEnabled
  });

  updateConnectionState(true);
  showToast(`Joined room ${roomId}`, 'success');
}

startRoomBtn.addEventListener('click', () => {
  const code = roomIdInput.value.trim() || generateRoomCode();
  roomIdInput.value = code;
  joinRoom(code, true);
});

joinRoomBtn.addEventListener('click', () => {
  const code = roomIdInput.value.trim();
  if (!code) {
    showToast('Please enter a room code first.', 'error');
    return;
  }
  joinRoom(code);
});

socket.on('connect', () => {
  showToast('Connected to signaling server.', 'success');
});

socket.on('room-joined', async data => {
  if (!roomId || data.room_id !== roomId) return;
  showToast('Room participants updated.', 'info');
  const roomOwner = data.room_owner;
  const username = localStorage.getItem('username');
  const shouldOffer = roomOwner === username;
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

socket.on('ice-candidate', async data => {
  if (!peerConnection || !data.candidate) return;
  try {
    await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
  } catch (err) {
    console.error('Ice candidate error', err);
  }
});

socket.on('interpreter-changed', data => {
  interpreterEnabled = !!data.enabled;
  interpreterSwitch.checked = interpreterEnabled;
  updateInterpreterLabel();
  showToast(`Interpreter mode ${interpreterEnabled ? 'enabled' : 'disabled'}`, 'info');
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

  localOverlay.width = localVideo.videoWidth || 1280;
  localOverlay.height = localVideo.videoHeight || 720;

  localCamera = new Camera(localVideo, {
    onFrame: async () => {
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
  const features = extractFeatures(results);
  if (!features || features.length !== 126) return;
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
    }
  } catch (err) {
    console.error('Prediction error', err);
  }
}

async function initApp() {
  updateInterpreterLabel();
  const valid = await verifyProfile();
  if (!valid) return;
  await fetchCallHistory();
  await startLocalMedia();
  await startDetection();
}

refreshHistoryBtn.addEventListener('click', fetchCallHistory);

initApp();
