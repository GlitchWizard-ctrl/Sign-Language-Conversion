// admin.js
(function () {
    const token = localStorage.getItem("authToken");
    const username = localStorage.getItem("username");
    const role = localStorage.getItem("role");

    if (!token || role !== "admin") {
        window.location.href = "login.html";
        return;
    }

    document.getElementById("adminUsernameLabel").textContent = username;

    // ---- Toast ----
    function toast(message, type = "info") {
        const container = document.getElementById("toastContainer");
        const el = document.createElement("div");
        el.className = `toast toast-${type}`;
        const icon = type === "success" ? "fa-circle-check" : type === "error" ? "fa-triangle-exclamation" : "fa-circle-info";
        el.innerHTML = `<i class="fa-solid ${icon}"></i><span>${message}</span>`;
        container.appendChild(el);
        setTimeout(() => el.remove(), 4000);
    }

    // ---- User dropdown ----
    const userPillBtn = document.getElementById("userPillBtn");
    const dropdownMenu = document.getElementById("userDropdownMenu");
    userPillBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        dropdownMenu.classList.toggle("show");
    });
    document.addEventListener("click", () => dropdownMenu.classList.remove("show"));

    document.getElementById("logoutBtn").addEventListener("click", async (e) => {
        e.preventDefault();
        await fetch("/api/logout", { method: "POST", headers: { "Authorization": `Bearer ${token}` } });
        localStorage.clear();
        window.location.href = "login.html";
    });

    // ---- Auth fetch wrapper ----
    async function authFetch(url, options = {}) {
        const res = await fetch(url, {
            ...options,
            headers: { ...(options.headers || {}), "Authorization": `Bearer ${token}` }
        });
        if (res.status === 401 || res.status === 403) {
            localStorage.clear();
            window.location.href = "login.html";
            throw new Error("unauthorized");
        }
        return res.json();
    }

    // ---- Stats ----
    async function loadStats() {
        try {
            const data = await authFetch("/api/admin/stats");
            if (!data.success) return;
            document.getElementById("statUsers").textContent = data.stats.total_users;
            document.getElementById("statCalls").textContent = data.stats.total_calls;
            document.getElementById("statActive").textContent = data.stats.active_calls;
            document.getElementById("statSamples").textContent = data.stats.total_samples ?? 0;
        } catch (e) { /* redirect already handled */ }
    }

    // ---- Users table ----
    async function loadUsers() {
        try {
            const data = await authFetch("/api/admin/users");
            if (!data.success) return;
            const tbody = document.getElementById("usersTableBody");
            if (data.users.length === 0) {
                tbody.innerHTML = `<tr><td colspan="5" class="history-empty">No users yet.</td></tr>`;
                return;
            }
            tbody.innerHTML = data.users.map(u => `
                <tr>
                    <td>${u.username}</td>
                    <td>${u.fullname}</td>
                    <td>${u.email}</td>
                    <td><span class="badge-pill">${u.role}</span></td>
                    <td>${new Date(u.created_at).toLocaleDateString()}</td>
                </tr>
            `).join("");
        } catch (e) {}
    }

    // ---- Calls table ----
    async function loadCalls() {
        try {
            const data = await authFetch("/api/admin/calls");
            if (!data.success) return;
            const tbody = document.getElementById("callsTableBody");
            if (data.calls.length === 0) {
                tbody.innerHTML = `<tr><td colspan="6" class="history-empty">No calls yet.</td></tr>`;
                return;
            }
            tbody.innerHTML = data.calls.map(c => {
                const status = c.ended_at
                    ? `<span class="status-pill status-pill-disconnected">Ended</span>`
                    : `<span class="status-pill status-pill-ready">Active</span>`;
                return `
                    <tr>
                        <td>${c.room_id}</td>
                        <td>${c.call_type}</td>
                        <td>${c.host_username}</td>
                        <td>${c.participants.join(", ")}</td>
                        <td>${new Date(c.started_at).toLocaleString()}</td>
                        <td>${status}</td>
                    </tr>`;
            }).join("");
        } catch (e) {}
    }

    // ---- Camera + sign capture ----
    const video = document.getElementById("preview");
    const canvas = document.getElementById("captureCanvas");
    const ctx = canvas.getContext("2d");
    const labelInput = document.getElementById("signLabel");
    const captureBtn = document.getElementById("captureBtn");
    const burstToggle = document.getElementById("burstToggle");
    const autoTrainToggle = document.getElementById("autoTrainToggle");
    const statusEl = document.getElementById("captureStatus");
    const autoTrainStatusEl = document.getElementById("autoTrainStatus");
    let burstInterval = null;

    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        } catch (e) {
            toast("Could not access camera.", "error");
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
            const data = await authFetch("/api/admin/samples", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ label, image })
            });
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
            scheduleAutoTrain(800); // burst just ended, train soon
        }
    });

    // ---- Sample counts table ----
    function renderCounts(counts) {
        const entries = Object.entries(counts);
        const tbody = document.getElementById("sampleCountsBody");
        const total = entries.reduce((sum, [, c]) => sum + c, 0);
        document.getElementById("sampleTotalBadge").textContent = `${total} total`;

        if (entries.length === 0) {
            tbody.innerHTML = `<tr><td colspan="3" class="history-empty">No signs recorded yet.</td></tr>`;
            return;
        }
        tbody.innerHTML = entries.map(([label, count]) => `
            <tr>
                <td>${label}</td>
                <td><span class="badge-pill">${count} samples</span></td>
                <td><button class="btn-danger table-btn-sm" data-label="${label}"><i class="fa-solid fa-trash"></i></button></td>
            </tr>
        `).join("");

        tbody.querySelectorAll(".btn-danger").forEach(btn => {
            btn.addEventListener("click", async () => {
                const label = btn.dataset.label;
                if (!confirm(`Delete all samples for '${label}'?`)) return;
                try {
                    const data = await authFetch(`/api/admin/samples/${encodeURIComponent(label)}`, { method: "DELETE" });
                    if (data.success) {
                        renderCounts(data.counts);
                        toast(`Deleted samples for '${label}'`, "success");
                    }
                } catch (e) {}
            });
        });
    }

    async function loadCounts() {
        try {
            const data = await authFetch("/api/admin/samples/stats");
            if (data.success) renderCounts(data.counts);
        } catch (e) {}
    }

    // ---- Train (manual + automatic share this) ----
    async function runTraining(isAuto) {
        const btn = document.getElementById("trainBtn");
        const resultEl = document.getElementById("trainResult");

        if (isAuto) {
            autoTrainStatusEl.textContent = "Auto-training on newly recorded samples...";
            autoTrainStatusEl.className = "capture-status ok";
        } else {
            btn.disabled = true;
        }

        const origBtnHTML = btn.innerHTML;
        if (!isAuto) btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Training...';
        if (!isAuto) resultEl.innerHTML = "";

        try {
            const data = await authFetch("/api/admin/train", { method: "POST" });
            if (data.success) {
                resultEl.innerHTML = `
                    <div class="train-success">
                        <p><strong>✓ Training complete${isAuto ? " (automatic)" : ""}.</strong></p>
                        <p>Accuracy: ${(data.accuracy * 100).toFixed(1)}% on held-out test samples</p>
                        <p>Trained on ${data.num_samples} samples across ${data.num_classes} signs</p>
                        <p>Classes: ${data.classes.join(", ")}</p>
                    </div>`;
                if (isAuto) {
                    autoTrainStatusEl.textContent = `✓ Auto-trained just now (${(data.accuracy * 100).toFixed(1)}% accuracy). Live model updated.`;
                    autoTrainStatusEl.className = "capture-status ok";
                } else {
                    toast("Model trained and live model updated.", "success");
                }
                loadStats();
            } else {
                if (isAuto) {
                    // Common during setup (e.g. "need at least 2 signs") — don't alarm the user, just show it quietly.
                    autoTrainStatusEl.textContent = `Auto-train skipped: ${data.message}`;
                    autoTrainStatusEl.className = "capture-status";
                } else {
                    resultEl.innerHTML = `<div class="train-error">✗ ${data.message}</div>`;
                    toast(data.message, "error");
                }
            }
        } catch (e) {
            if (isAuto) {
                autoTrainStatusEl.textContent = "✗ Auto-train failed: server unreachable.";
                autoTrainStatusEl.className = "capture-status error";
            } else {
                resultEl.innerHTML = `<div class="train-error">✗ Server unreachable.</div>`;
            }
        } finally {
            if (!isAuto) {
                btn.disabled = false;
                btn.innerHTML = origBtnHTML;
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

    document.getElementById("trainBtn").addEventListener("click", () => {
        cancelAutoTrain();
        runTraining(false);
    });

    loadStats();
    loadUsers();
    loadCalls();
    loadCounts();
})();