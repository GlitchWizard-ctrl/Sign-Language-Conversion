const authToken = localStorage.getItem('authToken');
if (!authToken) location.href = 'login.html';
document.getElementById('backBtn').addEventListener('click', () => location.href = 'index.html');

async function loadHistory() {
  try {
    const res = await fetch('/api/call-history', { headers: { 'Authorization': authToken } });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.message || 'Failed');
    const body = document.getElementById('historyBody');
    if (!data.history || !data.history.length) {
      body.innerHTML = '<tr><td colspan="6">No recent calls yet.</td></tr>';
      return;
    }
    body.innerHTML = data.history.map(h => `<tr><td>${h.room_id}</td><td>${h.caller}</td><td>${h.callee}</td><td>${h.interpreter_mode? 'Yes':'No'}</td><td>${h.duration_seconds}s</td><td>${h.created_at}</td></tr>`).join('');
  } catch (e) { console.error(e); document.getElementById('historyBody').innerHTML = '<tr><td colspan="6">Error loading history.</td></tr>'; }
}

loadHistory();
