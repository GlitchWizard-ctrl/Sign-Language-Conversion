document.getElementById('backBtn').addEventListener('click', () => location.href = 'index.html');
const lang = localStorage.getItem('language') || 'en';
document.getElementById('languageSelect').value = lang;
const hand = localStorage.getItem('hand') || 'right';
document.querySelectorAll('input[name=hand]').forEach(r => { if (r.value===hand) r.checked=true; });
document.getElementById('saveSettings').addEventListener('click', () => {
  const sel = document.getElementById('languageSelect').value;
  const handSel = document.querySelector('input[name=hand]:checked').value;
  localStorage.setItem('language', sel);
  localStorage.setItem('hand', handSel);
  alert('Settings saved');
  location.href = 'index.html';
});
