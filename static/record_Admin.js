// record_Admin.js
(function () {
    const token = localStorage.getItem("authToken");
    const role = localStorage.getItem("role");
    if (!token || role !== "admin") {
        window.location.href = "login.html";
        return;
    }

    document.getElementById("logoutBtn").addEventListener("click", async () => {
        await fetch("/api/logout", { method: "POST", headers: { "Authorization": `Bearer ${token}` } });
        localStorage.clear();
        window.location.href = "login.html";
    });

    const video = document.getElementById("preview");
    const canvas = document.getElementById("captureCanvas");
    const ctx = canvas.getContext("2d");
    const labelInput = document.getElementById("signLabel");
    const captureBtn = document.getElementById("captureBtn");
    const burstToggle = document.getElementById("burstToggle");
    const autoTrainToggle = document.getElementById("autoTrainToggle");
    const statusEl = document.getElementById("captureStatus");
    const autoTrainStatusEl = document.getElementById("autoTrainStatus");
    const countsEl = document.getElementById("sampleCounts");
    const trainBtn = document.getElementById("trainBtn");
    const trainResultEl = document.getElementById("trainResult");

    let burstInterval = null;

    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        } catch (e) {
            statusEl.textContent = "✗ Could not access camera.";
            statusEl.className = "capture-status error";
        }
    }
    startCamera();

    function grabFrameB64() {
        canvas.width = 320;
        canvas.height = 240;
        ctx.drawImage(video, 0, 0, 320, 240);
        return canvas.toDataURL("image/jpeg", 0.7);
    }

    async function captureOnce() {
        const label = labelInput.value.trim().toLowerCase();
        if (!label) {
            statusEl.textContent = "Enter a sign label first.";
            statusEl.className = "capture-status error";
            return;
        }
        const image = grabFrameB64();
        try {
            const res = await fetch("/api/admin/samples", {
                method: "POST",
                headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
                body: JSON.stringify({ label, image })
            });
            const data = await res.json();
            if (data.success) {
                statusEl.textContent = `✓ Captured '${label}' — ${data.count} samples so far.`;
                statusEl.className = "capture-status ok";
                renderCounts(data.counts);
                scheduleAutoTrain(3000);
            } else {
                statusEl.textContent = `✗ ${data.message}`;
                statusEl.className = "capture-status error";
            }
        } catch (e) {
            statusEl.textContent = "✗ Server unreachable.";
            statusEl.className = "capture-status error";
        }
    }

    captureBtn.addEventListener("click", captureOnce);

    burstToggle.addEventListener("change", () => {
        if (burstToggle.checked) {
            cancelAutoTrain();
            burstInterval = setInterval(captureOnce, 500);
        } else {
            clearInterval(burstInterval);
            burstInterval = null;
            scheduleAutoTrain(800);
        }
    });

    function renderCounts(counts) {
        const entries = Object.entries(counts);
        if (entries.length === 0) {
            countsEl.innerHTML = `<p class="muted">No signs recorded yet.</p>`;
            return;
        }
        countsEl.innerHTML = entries.map(([label, count]) => `
            <div class="count-row">
                <span class="count-label">${label}</span>
                <span class="count-badge">${count} samples</span>
                <button class="btn-danger-sm" data-label="${label}"><i class="fa-solid fa-trash"></i></button>
            </div>
        `).join("");

        countsEl.querySelectorAll(".btn-danger-sm").forEach(btn => {
            btn.addEventListener("click", async () => {
                const label = btn.dataset.label;
                if (!confirm(`Delete all samples for '${label}'?`)) return;
                const res = await fetch(`/api/admin/samples/${encodeURIComponent(label)}`, {
                    method: "DELETE",
                    headers: { "Authorization": `Bearer ${token}` }
                });
                const data = await res.json();
                if (data.success) renderCounts(data.counts);
            });
        });
    }

    async function loadCounts() {
        const res = await fetch("/api/admin/samples/stats", {
            headers: { "Authorization": `Bearer ${token}` }
        });
        const data = await res.json();
        if (data.success) renderCounts(data.counts);
    }
    loadCounts();

    // ---- Train (manual + automatic share this) ----
    async function runTraining(isAuto) {
        if (isAuto) {
            autoTrainStatusEl.textContent = "Auto-training on newly recorded samples...";
            autoTrainStatusEl.className = "capture-status ok";
        } else {
            trainBtn.disabled = true;
        }

        const origHTML = trainBtn.innerHTML;
        if (!isAuto) {
            trainBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Training...';
            trainResultEl.innerHTML = "";
        }

        try {
            const res = await fetch("/api/admin/train", {
                method: "POST",
                headers: { "Authorization": `Bearer ${token}` }
            });
            const data = await res.json();

            if (data.success) {
                trainResultEl.innerHTML = `
                    <div class="train-success">
                        <p><strong>✓ Training complete${isAuto ? " (automatic)" : ""}.</strong></p>
                        <p>Accuracy: ${(data.accuracy * 100).toFixed(1)}% on held-out test samples</p>
                        <p>Trained on ${data.num_samples} samples across ${data.num_classes} signs</p>
                        <p>Classes: ${data.classes.join(", ")}</p>
                        <p class="muted">Live captions in calls now use this model — no restart needed.</p>
                    </div>`;
                if (isAuto) {
                    autoTrainStatusEl.textContent = `✓ Auto-trained just now (${(data.accuracy * 100).toFixed(1)}% accuracy). Live model updated.`;
                    autoTrainStatusEl.className = "capture-status ok";
                }
            } else {
                if (isAuto) {
                    autoTrainStatusEl.textContent = `Auto-train skipped: ${data.message}`;
                    autoTrainStatusEl.className = "capture-status";
                } else {
                    trainResultEl.innerHTML = `<div class="train-error">✗ ${data.message}</div>`;
                }
            }
        } catch (e) {
            if (isAuto) {
                autoTrainStatusEl.textContent = "✗ Auto-train failed: server unreachable.";
                autoTrainStatusEl.className = "capture-status error";
            } else {
                trainResultEl.innerHTML = `<div class="train-error">✗ Server unreachable.</div>`;
            }
        } finally {
            if (!isAuto) {
                trainBtn.disabled = false;
                trainBtn.innerHTML = origHTML;
            }
        }
    }

    let autoTrainTimer = null;
    function scheduleAutoTrain(delayMs) {
        if (!autoTrainToggle.checked) return;
        cancelAutoTrain();
        autoTrainStatusEl.textContent = `Auto-training in ${Math.round(delayMs / 1000)}s if no more captures...`;
        autoTrainStatusEl.className = "capture-status";
        autoTrainTimer = setTimeout(() => runTraining(true), delayMs);
    }
    function cancelAutoTrain() {
        if (autoTrainTimer) {
            clearTimeout(autoTrainTimer);
            autoTrainTimer = null;
        }
    }

    trainBtn.addEventListener("click", () => {
        cancelAutoTrain();
        runTraining(false);
    });
})();