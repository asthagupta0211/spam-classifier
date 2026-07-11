async function predict() {
    const messageEl = document.getElementById('message');
    const resultEl = document.getElementById('result');
    const btn = document.getElementById('predictBtn');
    const message = messageEl.value.trim();

    resultEl.className = 'result';
    resultEl.innerHTML = '';

    if (!message) {
        resultEl.className = 'result error';
        resultEl.textContent = 'Please enter a message.';
        return;
    }

    btn.disabled = true;
    btn.classList.add('loading');

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const data = await res.json();

        if (!res.ok) {
            resultEl.className = 'result error';
            resultEl.textContent = data.error || 'Something went wrong.';
            return;
        }

        const isSpam = data.result === 'Spam';
        resultEl.className = 'result ' + (isSpam ? 'spam' : 'ham');

        const dot = document.createElement('span');
        dot.className = 'result-dot';

        const label = document.createElement('span');
        label.textContent = data.result;

        resultEl.appendChild(dot);
        resultEl.appendChild(label);

        // Confidence bar renders only if the backend provides a score;
        // safe no-op otherwise, keeping this compatible with the existing API.
        if (typeof data.confidence === 'number') {
            const track = document.createElement('div');
            track.className = 'confidence-track';
            const fill = document.createElement('div');
            fill.className = 'confidence-fill';
            fill.style.width = Math.round(data.confidence * 100) + '%';
            track.appendChild(fill);
            resultEl.appendChild(track);
        }
    } catch (err) {
        resultEl.className = 'result error';
        resultEl.textContent = 'Server error. Is the backend running?';
    } finally {
        btn.disabled = false;
        btn.classList.remove('loading');
    }
}

document.getElementById('message').addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        predict();
    }
});