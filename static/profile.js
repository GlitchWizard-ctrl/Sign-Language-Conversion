const authToken = localStorage.getItem('authToken');
if (!authToken) window.location.href = 'login.html';

const fullnameEl = document.getElementById('fullname');
const emailEl = document.getElementById('email');
const passwordEl = document.getElementById('password');
const saveBtn = document.getElementById('saveBtn');
const cancelBtn = document.getElementById('cancelBtn');
const backBtn = document.getElementById('backBtn');

function showToast(msg) {
  const t = document.createElement('div'); t.className = 'toast'; t.textContent = msg; document.body.appendChild(t);
  setTimeout(() => t.remove(), 2500);
}

async function loadProfile() {
  try {
    const res = await fetch('/api/profile', { headers: { 'Authorization': `Bearer ${authToken}` } });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error('Not authorized');
    fullnameEl.value = data.fullname || '';
    emailEl.value = data.email || '';
  } catch (e) { showToast('Could not load profile'); console.error(e); }
}

saveBtn.addEventListener('click', async () => {
  const payload = { fullname: fullnameEl.value.trim(), email: emailEl.value.trim() };
  if (passwordEl.value) payload.password = passwordEl.value;
  try {
    const res = await fetch('/api/profile', { method: 'PUT', headers: { 'Content-Type':'application/json', 'Authorization': `Bearer ${authToken}` }, body: JSON.stringify(payload) });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.message || 'Failed');
    showToast('Profile updated');
    setTimeout(() => location.href = 'index.html', 800);
  } catch (e) { showToast('Update failed'); console.error(e); }
});

cancelBtn.addEventListener('click', () => location.href = 'index.html');
backBtn.addEventListener('click', () => location.href = 'index.html');

loadProfile();
