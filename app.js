// ============================================================
// AUTH CHECK
// ============================================================
const authToken = localStorage.getItem("authToken");
if (!authToken) {
    window.location.href = "login.html";
}

document.addEventListener("DOMContentLoaded", () => {

    // Set username
    const username = localStorage.getItem("username") || "User";
    document.getElementById("userNameLabel").textContent = username;

    // Logout
    document.getElementById("logoutBtn").addEventListener("click", async () => {
        try {
            await fetch("/api/logout", {
                method: "POST",
                headers: { "Authorization": authToken }
            });
        } catch (err) {
            console.error("Logout error:", err);
        } finally {
            localStorage.removeItem("authToken");
            localStorage.removeItem("username");
            window.location.href = "login.html";
        }
    });

    // ============================================================
    // TAB NAVIGATION
    // ============================================================
    const tabs = document.querySelectorAll(".nav-tab");
    const panels = document.querySelectorAll(".tab-panel");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetTab = tab.getAttribute("data-tab");
            tabs.forEach(t => t.classList.remove("active"));
            panels.forEach(p => p.classList.remove("active"));
            tab.classList.add("active");
            document.getElementById(targetTab).classList.add("active");
            handleTabSwitch(targetTab);
        });
    });

    function handleTabSwitch(tabName) {
        if (tabName === "signToAudioTab") {
            stopS2ACamera();
        } else if (tabName === "audioToSignTab") {
            stopS2ACamera();
            stopA2SListening();
        }
    }

    // ============================================================
    // MEDIAPIPE HANDS SETUP
    // ============================================================
    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.6
    });

    const HAND_CONNECTIONS = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [5,9],[9,10],[10,11],[11,12],
        [9,13],[13,14],[14,15],[15,16],
        [13,17],[0,17],[17,18],[18,19],[19,20]
    ];

    // ============================================================
    // TOAST NOTIFICATIONS
    // ============================================================
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        let icon = "fa-info-circle";
        if (type === "success") icon = "fa-check-circle";
        else if (type === "error") icon = "fa-exclamation-circle";
        toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${message}</span>`;
        document.getElementById("toastContainer").appendChild(toast);
        setTimeout(() => {
            toast.style.animation = "slideIn 0.3s ease reverse";
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // ============================================================
    // DRAW HAND LANDMARKS
    // ============================================================
    function drawHand(ctx, landmarks, label) {
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;

        ctx.strokeStyle = label === "Left" ? "#6366f1" : "#a855f7";
        ctx.lineWidth = 4;
        ctx.shadowColor = "rgba(0,0,0,0.5)";
        ctx.shadowBlur = 4;

        HAND_CONNECTIONS.forEach(([s, e]) => {
            const sp = landmarks[s], ep = landmarks[e];
            if (sp && ep) {
                ctx.beginPath();
                ctx.moveTo(sp.x * w, sp.y * h);
                ctx.lineTo(ep.x * w, ep.y * h);
                ctx.stroke();
            }
        });

        landmarks.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x * w, pt.y * h, 6, 0, 2 * Math.PI);
            ctx.fillStyle = "#ffffff";
            ctx.fill();
            ctx.strokeStyle = "#1f2937";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });

        const wrist = landmarks[0];
        if (wrist) {
            ctx.shadowBlur = 0;
            ctx.fillStyle = label === "Left" ? "#10b981" : "#a855f7";
            ctx.font = "bold 20px Outfit";
            ctx.fillText(label, (wrist.x * w) - 20, (wrist.y * h) - 20);
        }
    }

    // ============================================================
    // EXTRACT LANDMARK FEATURES
    // ============================================================
    function extractLandmarks(results) {
        let left = new Array(63).fill(0.0);
        let right = new Array(63).fill(0.0);

        if (results.multiHandLandmarks && results.multiHandedness) {
            for (let i = 0; i < results.multiHandLandmarks.length; i++) {
                const handLm = results.multiHandLandmarks[i];
                const handedness = results.multiHandedness[i];
                const label = handedness.label ||
                    (handedness.classification && handedness.classification[0].label);
                let coords = [];
                for (const lm of handLm) coords.push(lm.x, lm.y, lm.z);
                if (label === "Left") left = coords;
                else if (label === "Right") right = coords;
            }
        }
        return left.concat(right);
    }

    // ============================================================
    // CAMERA UTILITIES
    // ============================================================
    let isWebcamRunning = false;
    let cameraInstance = null;
    let currentResults = null;

    async function initCamera(videoEl, canvasEl, noHandEl, leftBadge, rightBadge, onResults) {
        if (isWebcamRunning) return;
        const ctx = canvasEl.getContext("2d");
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

        let leftDetected = false;
        let rightDetected = false;

        hands.onResults((results) => {
            currentResults = results;
            ctx.save();
            ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
            ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);
            ctx.restore();

            leftDetected = false;
            rightDetected = false;

            if (results.multiHandLandmarks && results.multiHandedness) {
                for (let i = 0; i < results.multiHandLandmarks.length; i++) {
                    const label = results.multiHandedness[i].label ||
                        (results.multiHandedness[i].classification &&
                         results.multiHandedness[i].classification[0].label);
                    if (label === "Left") leftDetected = true;
                    if (label === "Right") rightDetected = true;
                    drawHand(ctx, results.multiHandLandmarks[i], label);
                }
            }

            noHandEl.style.display = (!leftDetected && !rightDetected) ? "block" : "none";
            leftBadge.className  = `hand-badge ${leftDetected  ? "active" : ""}`;
            rightBadge.className = `hand-badge ${rightDetected ? "active" : ""}`;

            if (onResults) onResults(results, leftDetected, rightDetected);
        });

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
            videoEl.srcObject = stream;
            canvasEl.width = 1280;
            canvasEl.height = 720;

            cameraInstance = new Camera(videoEl, {
                onFrame: async () => {
                    if (isWebcamRunning) await hands.send({ image: videoEl });
                },
                width: 1280,
                height: 720
            });

            await cameraInstance.start();
            isWebcamRunning = true;
        } catch (err) {
            console.error("Camera start failed:", err);
            showToast("Could not access camera. Please grant permission.", "error");
        }
    }

    function stopCamera(videoEl) {
        if (!isWebcamRunning) return;
        isWebcamRunning = false;
        if (cameraInstance) { cameraInstance.stop(); cameraInstance = null; }
        if (videoEl && videoEl.srcObject) {
            videoEl.srcObject.getTracks().forEach(t => t.stop());
            videoEl.srcObject = null;
        }
    }

    // ============================================================
    // TAB 1: SIGN LANGUAGE → AUDIO
    // ============================================================
    let s2aRunning = false;
    let lastPredictTime = 0;
    let lastSpokenSign = "";
    let lastSpeakTime = 0;
    const PREDICT_THROTTLE_MS = 200;
    const SPEAK_COOLDOWN_MS = 2500;

    const s2aStartBtn   = document.getElementById("s2aStartBtn");
    const s2aStopBtn    = document.getElementById("s2aStopBtn");
    const s2aClearBtn   = document.getElementById("s2aClearBtn");
    const s2aSpeakBtn   = document.getElementById("s2aSpeakBtn");
    const s2aResultOverlay = document.getElementById("s2aResultOverlay");
    const s2aSignVal    = document.getElementById("s2aSignVal");
    const s2aConfVal    = document.getElementById("s2aConfVal");
    const s2aCurrentSignText = document.getElementById("s2aCurrentSignText");
    const s2aCurrentSignConf = document.getElementById("s2aCurrentSignConf");
    const s2aTranscript = document.getElementById("s2aTranscript");
    const s2aSpeedSlider = document.getElementById("s2aSpeedSlider");
    const s2aSpeedVal   = document.getElementById("s2aSpeedVal");
    const s2aVoiceSelect = document.getElementById("s2aVoiceSelect");
    const s2aAutoSpeak  = document.getElementById("s2aAutoSpeak");
    const s2aModelStatusDot  = document.getElementById("s2aModelStatusDot");
    const s2aModelStatusText = document.getElementById("s2aModelStatusText");

    let currentDetectedSign = "";

    // Populate TTS voices
    function loadVoices() {
        const voices = window.speechSynthesis.getVoices();
        s2aVoiceSelect.innerHTML = '<option value="">Default Voice</option>';
        voices.forEach((v, i) => {
            const opt = document.createElement("option");
            opt.value = i;
            opt.textContent = `${v.name} (${v.lang})`;
            s2aVoiceSelect.appendChild(opt);
        });
    }
    window.speechSynthesis.onvoiceschanged = loadVoices;
    loadVoices();

    s2aSpeedSlider.addEventListener("input", () => {
        s2aSpeedVal.textContent = parseFloat(s2aSpeedSlider.value).toFixed(1) + "x";
    });

    // Check model status on load
    async function checkModelStatus() {
        try {
            const res = await fetch("/api/status", { headers: { "Authorization": authToken } });
            const data = await res.json();
            if (data.model_trained) {
                s2aModelStatusDot.style.color = "var(--success)";
                s2aModelStatusText.textContent = "Model Ready — " + (data.classes || []).length + " signs loaded";
            } else {
                s2aModelStatusDot.style.color = "var(--error)";
                s2aModelStatusText.textContent = "No trained model found";
            }
        } catch (e) {
            s2aModelStatusText.textContent = "Could not connect to server";
        }
    }
    checkModelStatus();

    function speakText(text) {
        if (!text || !window.speechSynthesis) return;
        window.speechSynthesis.cancel();
        const utt = new SpeechSynthesisUtterance(text);
        utt.rate = parseFloat(s2aSpeedSlider.value);
        const voices = window.speechSynthesis.getVoices();
        const selectedIdx = s2aVoiceSelect.value;
        if (selectedIdx !== "" && voices[parseInt(selectedIdx)]) {
            utt.voice = voices[parseInt(selectedIdx)];
        }
        window.speechSynthesis.speak(utt);
    }

    function addToTranscript(sign, confidence) {
        const existing = s2aTranscript.querySelector(".transcript-placeholder");
        if (existing) existing.remove();
        const span = document.querySelector("span[style*='color: var(--text-muted)']");

        // Remove placeholder text on first entry
        if (s2aTranscript.innerHTML.includes("Signs spoken will appear here")) {
            s2aTranscript.innerHTML = "";
        }

        const chip = document.createElement("div");
        chip.className = "transcript-chip";
        chip.innerHTML = `<span class="chip-sign">${sign}</span><span class="chip-conf">${confidence}</span>`;
        s2aTranscript.appendChild(chip);
        s2aTranscript.scrollTop = s2aTranscript.scrollHeight;
    }

    s2aSpeakBtn.addEventListener("click", () => {
        if (currentDetectedSign) speakText(currentDetectedSign);
    });

    s2aClearBtn.addEventListener("click", () => {
        s2aTranscript.innerHTML = '<span style="color: var(--text-muted); font-size:13px;">Signs spoken will appear here...</span>';
        s2aCurrentSignText.textContent = "Waiting for sign...";
        s2aCurrentSignConf.textContent = "";
        currentDetectedSign = "";
        s2aSpeakBtn.disabled = true;
    });

    s2aStartBtn.addEventListener("click", async function () {
        this.classList.add("btn-disabled");
        this.disabled = true;

        await initCamera(
            document.getElementById("s2aVideo"),
            document.getElementById("s2aCanvas"),
            document.getElementById("s2aNoHandWarning"),
            document.getElementById("s2aLeftBadge"),
            document.getElementById("s2aRightBadge"),
            handleS2AFrame
        );

        s2aStopBtn.classList.remove("btn-disabled");
        s2aStopBtn.disabled = false;
        s2aRunning = true;
        s2aResultOverlay.style.display = "flex";
        showToast("Camera started. Show your hand signs!", "success");
    });

    s2aStopBtn.addEventListener("click", function () {
        stopS2ACamera();
        showToast("Camera stopped.", "info");
    });

    function stopS2ACamera() {
        if (!s2aRunning) return;
        s2aRunning = false;
        stopCamera(document.getElementById("s2aVideo"));
        s2aStartBtn.classList.remove("btn-disabled");
        s2aStartBtn.disabled = false;
        s2aStopBtn.classList.add("btn-disabled");
        s2aStopBtn.disabled = true;
        s2aResultOverlay.style.display = "none";
    }

    async function handleS2AFrame(results, leftDetected, rightDetected) {
        if (!s2aRunning) return;
        const now = Date.now();
        if (now - lastPredictTime < PREDICT_THROTTLE_MS) return;
        if (!leftDetected && !rightDetected) {
            s2aSignVal.textContent = "—";
            s2aConfVal.textContent = "0%";
            return;
        }

        lastPredictTime = now;
        const features = extractLandmarks(results);

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json", "Authorization": authToken },
                body: JSON.stringify({ features })
            });
            const data = await res.json();
            if (res.ok && data.success) {
                const sign = `Sign ${data.predicted_class}`;
                const conf = Math.round(data.confidence * 100);
                const confStr = `${conf}%`;

                s2aSignVal.textContent = sign;
                s2aConfVal.textContent = confStr;
                s2aCurrentSignText.textContent = sign;
                s2aCurrentSignConf.textContent = `Confidence: ${confStr}`;
                currentDetectedSign = sign;
                s2aSpeakBtn.disabled = false;

                // Auto-speak if enabled, high confidence, and cooldown passed
                if (s2aAutoSpeak.checked && conf >= 70 && data.predicted_class !== lastSpokenSign
                    && (now - lastSpeakTime > SPEAK_COOLDOWN_MS)) {
                    speakText(sign);
                    addToTranscript(sign, confStr);
                    lastSpokenSign = data.predicted_class;
                    lastSpeakTime = now;
                }
            }
        } catch (err) {
            console.error("Prediction error:", err);
        }
    }

    // ============================================================
    // TAB 2: AUDIO → SIGN LANGUAGE
    // ============================================================
    let recognition = null;
    let a2sRunning = false;

    const a2sStartBtn   = document.getElementById("a2sStartBtn");
    const a2sStopBtn    = document.getElementById("a2sStopBtn");
    const a2sClearBtn   = document.getElementById("a2sClearBtn");
    const a2sInterimText = document.getElementById("a2sInterimText");
    const a2sTranscript = document.getElementById("a2sTranscript");
    const a2sSignOutput = document.getElementById("a2sSignOutput");
    const a2sMicStatus  = document.getElementById("a2sMicStatus");
    const a2sMicHint    = document.getElementById("a2sMicHint");
    const a2sMicIconWrap = document.getElementById("a2sMicIconWrap");
    const a2sLangSelect = document.getElementById("a2sLangSelect");

    // ISL sign keyword mapping
    const ISL_SIGN_MAP = {
        "hello": { emoji: "👋", description: "Wave hand side to side" },
        "hi": { emoji: "👋", description: "Wave hand side to side" },
        "thank": { emoji: "🙏", description: "Both palms together, move forward" },
        "thanks": { emoji: "🙏", description: "Both palms together, move forward" },
        "you": { emoji: "👉", description: "Point index finger toward the person" },
        "i": { emoji: "👆", description: "Point index finger to yourself" },
        "me": { emoji: "👆", description: "Point index finger to yourself" },
        "yes": { emoji: "✊", description: "Fist bobbing up and down" },
        "no": { emoji: "✌️", description: "Index and middle finger wag side to side" },
        "help": { emoji: "🤝", description: "Fist on flat palm, move upward" },
        "please": { emoji: "🤲", description: "Flat hand circles on chest" },
        "sorry": { emoji: "✊", description: "Fist circles on chest" },
        "good": { emoji: "👍", description: "Flat hand from chin moving outward" },
        "bad": { emoji: "👎", description: "Hand flips downward from chin" },
        "name": { emoji: "🤟", description: "Two fingers tap on opposite two fingers" },
        "what": { emoji: "🤷", description: "Fingers spread, hands shrug" },
        "where": { emoji: "☝️", description: "Index finger wagging side to side" },
        "how": { emoji: "🙌", description: "Hands back-to-back, roll forward" },
        "when": { emoji: "🕐", description: "Index finger circles on opposite wrist" },
        "who": { emoji: "☝️", description: "Index finger at lips moving in a circle" },
        "water": { emoji: "💧", description: "W-hand shape taps chin" },
        "eat": { emoji: "🍽️", description: "Flat O-hand taps mouth" },
        "drink": { emoji: "🥛", description: "C-hand tilts toward mouth" },
        "home": { emoji: "🏠", description: "Flat O-hand taps cheek twice" },
        "work": { emoji: "💼", description: "S-fists, dominant taps on other" },
        "school": { emoji: "🏫", description: "Flat hands clap together twice" },
        "love": { emoji: "❤️", description: "Cross arms over chest" },
        "friend": { emoji: "🤝", description: "Hook index fingers, reverse" },
        "family": { emoji: "👨‍👩‍👧‍👦", description: "F-hands circle outward" },
        "mother": { emoji: "👩", description: "Open hand, thumb touches chin" },
        "father": { emoji: "👨", description: "Open hand, thumb touches forehead" },
        "happy": { emoji: "😊", description: "Flat hand brushes up chest twice" },
        "sad": { emoji: "😢", description: "Both hands slide down face" },
        "come": { emoji: "🫵", description: "Index finger beckons inward" },
        "go": { emoji: "➡️", description: "Index fingers point and arc forward" },
        "stop": { emoji: "🛑", description: "Flat hand chops on flat palm" },
        "more": { emoji: "➕", description: "Flat O-hands tap fingertips together" },
        "again": { emoji: "🔄", description: "Bent hand arcs up onto flat palm" },
        "understand": { emoji: "💡", description: "Index finger flicks up from fist near head" },
        "know": { emoji: "🧠", description: "Flat hand taps side of forehead" },
        "wait": { emoji: "✋", description: "Both hands wiggle fingers facing out" },
        "book": { emoji: "📖", description: "Flat hands open like a book" },
        "time": { emoji: "⌚", description: "Index finger taps back of wrist" },
        "money": { emoji: "💰", description: "Flat O-hand taps flat palm" },
        "food": { emoji: "🍎", description: "Flat O-hand taps mouth" },
        "sleep": { emoji: "😴", description: "Hand lowers in front of face, head tilts" },
        "day": { emoji: "☀️", description: "Index finger arcs from elbow to upright" },
        "night": { emoji: "🌙", description: "Bent hand arcs down over other arm" },
        "big": { emoji: "⬛", description: "L-hands pull apart" },
        "small": { emoji: "⬜", description: "Flat hands press together" },
        "hot": { emoji: "🔥", description: "Claw hand at mouth twists away" },
        "cold": { emoji: "🥶", description: "Fists shiver near shoulders" },
    };

    function getSignForWord(word) {
        const clean = word.toLowerCase().replace(/[^a-z]/g, "");
        return ISL_SIGN_MAP[clean] || { emoji: "🤟", description: `Fingerspell: ${word.toUpperCase()}` };
    }

    function renderSignCards(sentence) {
        const words = sentence.trim().split(/\s+/).filter(Boolean);
        if (!words.length) return;

        // Clear placeholder
        if (a2sSignOutput.querySelector(".sign-output-placeholder")) {
            a2sSignOutput.innerHTML = "";
        }

        words.forEach(word => {
            const signInfo = getSignForWord(word);
            const card = document.createElement("div");
            card.className = "sign-card";
            card.innerHTML = `
                <div class="sign-card-emoji">${signInfo.emoji}</div>
                <div class="sign-card-word">${word}</div>
                <div class="sign-card-desc">${signInfo.description}</div>
            `;
            a2sSignOutput.appendChild(card);
        });

        a2sSignOutput.scrollTop = a2sSignOutput.scrollHeight;
    }

    function addA2STranscriptLine(text) {
        if (a2sTranscript.innerHTML.includes("Recognized sentences")) {
            a2sTranscript.innerHTML = "";
        }
        const line = document.createElement("div");
        line.className = "transcript-line";
        line.innerHTML = `<span class="transcript-time">${new Date().toLocaleTimeString()}</span> ${text}`;
        a2sTranscript.appendChild(line);
        a2sTranscript.scrollTop = a2sTranscript.scrollHeight;
    }

    function setupRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            showToast("Speech recognition is not supported in this browser. Use Chrome.", "error");
            return null;
        }

        const rec = new SpeechRecognition();
        rec.continuous = true;
        rec.interimResults = true;
        rec.lang = a2sLangSelect.value;

        rec.onstart = () => {
            a2sMicStatus.innerHTML = '<i class="fa-solid fa-microphone"></i> Listening...';
            a2sMicStatus.classList.add("active");
            a2sMicIconWrap.classList.add("pulsing");
            a2sMicHint.textContent = "Listening... speak clearly";
        };

        rec.onresult = (event) => {
            let interim = "";
            let final = "";

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    final += transcript;
                } else {
                    interim += transcript;
                }
            }

            a2sInterimText.textContent = interim || final || "";
            a2sInterimText.style.color = interim ? "var(--text-muted)" : "var(--text-main)";
            a2sInterimText.style.fontStyle = interim ? "italic" : "normal";

            if (final) {
                renderSignCards(final);
                addA2STranscriptLine(final);
                a2sInterimText.textContent = "";
            }
        };

        rec.onerror = (event) => {
            if (event.error !== "no-speech") {
                showToast(`Mic error: ${event.error}`, "error");
            }
        };

        rec.onend = () => {
            if (a2sRunning) rec.start(); // keep alive
        };

        return rec;
    }

    a2sStartBtn.addEventListener("click", function () {
        recognition = setupRecognition();
        if (!recognition) return;

        this.classList.add("btn-disabled");
        this.disabled = true;
        a2sStopBtn.classList.remove("btn-disabled");
        a2sStopBtn.disabled = false;
        a2sRunning = true;
        recognition.lang = a2sLangSelect.value;
        recognition.start();
        showToast("Microphone started. Speak now!", "success");
    });

    a2sStopBtn.addEventListener("click", () => stopA2SListening());

    function stopA2SListening() {
        if (!a2sRunning) return;
        a2sRunning = false;
        if (recognition) { recognition.stop(); recognition = null; }
        a2sStartBtn.classList.remove("btn-disabled");
        a2sStartBtn.disabled = false;
        a2sStopBtn.classList.add("btn-disabled");
        a2sStopBtn.disabled = true;
        a2sMicStatus.innerHTML = '<i class="fa-solid fa-microphone-slash"></i> Mic Off';
        a2sMicStatus.classList.remove("active");
        a2sMicIconWrap.classList.remove("pulsing");
        a2sMicHint.textContent = 'Press "Start Listening" and speak clearly';
    }

    a2sClearBtn.addEventListener("click", () => {
        a2sSignOutput.innerHTML = `
            <div class="sign-output-placeholder">
                <i class="fa-solid fa-hand-sparkles" style="font-size:32px; color:var(--primary); margin-bottom:12px;"></i>
                <p>Speak something and the ISL signs<br>will appear here word by word</p>
            </div>`;
        a2sTranscript.innerHTML = '<span style="color: var(--text-muted); font-size:13px;">Recognized sentences will appear here...</span>';
        a2sInterimText.textContent = "Your speech will appear here in real-time...";
        a2sInterimText.style.color = "var(--text-muted)";
        a2sInterimText.style.fontStyle = "italic";
    });

});
