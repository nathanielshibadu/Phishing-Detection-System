const checkBtn = document.getElementById('checkBtn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');

function setStatus(t) { statusEl.textContent = t; }
function setResultHTML(html) { resultEl.innerHTML = html; }

checkBtn.addEventListener('click', async () => {
  setStatus('Getting current tab URL...');
  setResultHTML('');
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab && tab.url ? tab.url : null;
    if (!url) {
      setStatus('Could not get current URL');
      return;
    }
    setStatus('Contacting local PhishGuard API...');
    const res = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: url })
    });
    if (!res.ok) {
      const text = await res.text();
      setStatus('Server returned error: ' + res.status);
      setResultHTML('<div class="error">' + text + '</div>');
      return;
    }
    const data = await res.json();
    setStatus('Result received');
    let html = `<strong>Label:</strong> ${data.label} <br/>
                <strong>Confidence:</strong> ${(data.confidence*100).toFixed(1)}% <br/>
                <details><summary>Probabilities</summary><pre>${JSON.stringify(data.probs, null, 2)}</pre></details>`;
    setResultHTML(html);
  } catch (err) {
    setStatus('Error: ' + err.message);
    setResultHTML('<div class="error">' + err.message + '</div>');
    console.error(err);
  }
});
