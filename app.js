// Check authentication on startup
const authToken = localStorage.getItem("authToken");
if (!authToken) {
    window.location.href = "login.html";
}

document.addEventListener("DOMContentLoaded", () => {
    // Set username label
    const username = localStorage.getItem("username") || "Admin User";
    document.getElementById("userNameLabel").textContent = username;

    // Logout handling
    document.getElementById("logoutBtn").addEventListener("click", () => {
        localStorage.removeItem("authToken");
        localStorage.removeItem("username");
        window.location.href = "login.html";
    });

    // ==========================================================================
    // TAB NAVIGATION
    // ==========================================================================
    const tabs = document.querySelectorAll(".nav-tab");
    const panels = document.querySelectorAll(".tab-panel");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetTab = tab.getAttribute("data-tab");
            
            // Switch tabs
            tabs.forEach(t => t.classList.remove("active"));
            panels.forEach(p => p.classList.remove("active"));
            
            tab.classList.add("active");
            document.getElementById(targetTab).classList.add("active");

            // Handle Camera switching
            handleTabSwitch(targetTab);
        });
    });

    // ==========================================================================
    // CONFIG & STATE
    // ==========================================================================
    let activeMode = "record"; // "record", "train", "predict"
    let isWebcamRunning = false;
    let cameraInstance = null;
    let currentResults = null; // Store latest landmarks

    // Recording State
    const SAMPLES_PER_SIGN = 2;
    let collectedSamplesCount = 0;
    let currentSignToRecord = "1";
    let isLeftHandDetected = false;
    let isRightHandDetected = false;

    // UI Elements
    const toastContainer = document.getElementById("toastContainer");

    // ==========================================================================
    // MEDIAPIPE HANDS CLIENT SETUP
    // ==========================================================================
    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.6
    });

    // Custom skeleton connections for rendering
    const HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
    ];

    // ==========================================================================
    // NOTIFICATION SYSTEM
    // ==========================================================================
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        
        let icon = "fa-info-circle";
        if (type === "success") icon = "fa-check-circle";
        else if (type === "error") icon = "fa-exclamation-circle";

        toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${message}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = "slideIn 0.3s ease reverse";
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // ==========================================================================
    // DRAWING HAND LANDMARKS
    // ==========================================================================
    function drawHand(ctx, landmarks, label) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;

        // Draw connections
        ctx.strokeStyle = label === "Left" ? "#6366f1" : "#a855f7"; // Indigo for left, purple for right
        ctx.lineWidth = 4;
        ctx.shadowColor = "rgba(0,0,0,0.5)";
        ctx.shadowBlur = 4;

        HAND_CONNECTIONS.forEach(([start, end]) => {
            const startPt = landmarks[start];
            const endPt = landmarks[end];
            if (startPt && endPt) {
                ctx.beginPath();
                ctx.moveTo(startPt.x * width, startPt.y * height);
                ctx.lineTo(endPt.x * width, endPt.y * height);
                ctx.stroke();
            }
        });

        // Draw key points
        landmarks.forEach((pt) => {
            ctx.beginPath();
            ctx.arc(pt.x * width, pt.y * height, 6, 0, 2 * Math.PI);
            ctx.fillStyle = "#ffffff";
            ctx.fill();
            ctx.strokeStyle = "#1f2937";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });

        // Label above wrist (point 0)
        const wrist = landmarks[0];
        if (wrist) {
            ctx.shadowBlur = 0;
            ctx.fillStyle = label === "Left" ? "#10b981" : "#a855f7";
            ctx.font = "bold 20px Outfit";
            ctx.fillText(label, (wrist.x * width) - 20, (wrist.y * height) - 20);
        }
    }

    // ==========================================================================
    // EXTRACT & FORMAT FEATURE COORDINATES
    // ==========================================================================
    function extractLandmarks(results) {
        let left = new Array(63).fill(0.0);
        let right = new Array(63).fill(0.0);

        if (results.multiHandLandmarks && results.multiHandedness) {
            for (let i = 0; i < results.multiHandLandmarks.length; i++) {
                const handLm = results.multiHandLandmarks[i];
                const handedness = results.multiHandedness[i];
                
                const label = handedness.label || (handedness.classification && handedness.classification[0].label);
                
                let coords = [];
                for (const lm of handLm) {
                    coords.push(lm.x, lm.y, lm.z);
                }
                
                if (label === "Left") {
                    left = coords;
                } else if (label === "Right") {
                    right = coords;
                }
            }
        }
        return left.concat(right);
    }

    // ==========================================================================
    // WEBCAM PROCESSOR FOR MEDIA PIPE
    // ==========================================================================
    async function initCamera(videoEl, canvasEl, noHandWarningEl, leftBadge, rightBadge, processCallback) {
        if (isWebcamRunning) return;
        
        const ctx = canvasEl.getContext("2d");

        // Clear canvas
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

        // Process hands listener
        hands.onResults((results) => {
            currentResults = results;
            
            // Mirror camera frame on canvas
            ctx.save();
            ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
            ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);
            ctx.restore();

            isLeftHandDetected = false;
            isRightHandDetected = false;

            if (results.multiHandLandmarks && results.multiHandedness) {
                for (let i = 0; i < results.multiHandLandmarks.length; i++) {
                    const label = results.multiHandedness[i].label || 
                                  (results.multiHandedness[i].classification && results.multiHandedness[i].classification[0].label);
                    
                    if (label === "Left") isLeftHandDetected = true;
                    if (label === "Right") isRightHandDetected = true;

                    drawHand(ctx, results.multiHandLandmarks[i], label);
                }
            }

            // Hand warning visual state
            if (!isLeftHandDetected && !isRightHandDetected) {
                noHandWarningEl.style.display = "block";
            } else {
                noHandWarningEl.style.display = "none";
            }

            // Badges indicators
            leftBadge.className = `hand-badge ${isLeftHandDetected ? 'active' : ''}`;
            rightBadge.className = `hand-badge ${isRightHandDetected ? 'active' : ''}`;

            // Optional prediction processing
            if (processCallback) {
                processCallback(results);
            }
        });

        // Setup camera utilities
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 1280, height: 720 }
            });
            videoEl.srcObject = stream;
            
            canvasEl.width = 1280;
            canvasEl.height = 720;

            cameraInstance = new Camera(videoEl, {
                onFrame: async () => {
                    if (isWebcamRunning) {
                        await hands.send({ image: videoEl });
                    }
                },
                width: 1280,
                height: 720
            });
            
            await cameraInstance.start();
            isWebcamRunning = true;
        } catch (err) {
            console.error("Camera start failed:", err);
            showToast("Could not access web camera. Please grant permissions.", "error");
        }
    }

    function stopCamera(videoEl) {
        if (!isWebcamRunning) return;
        
        isWebcamRunning = false;
        if (cameraInstance) {
            cameraInstance.stop();
            cameraInstance = null;
        }
        if (videoEl.srcObject) {
            videoEl.srcObject.getTracks().forEach(track => track.stop());
            videoEl.srcObject = null;
        }
    }

    function handleTabSwitch(tabName) {
        // Stop any running camera instance first
        stopCamera(document.getElementById("recordVideo"));
        stopCamera(document.getElementById("predictVideo"));
        
        isWebcamRunning = false;
        
        // Reset camera button states
        document.getElementById("startRecordCamBtn").classList.remove("btn-disabled");
        document.getElementById("startRecordCamBtn").disabled = false;
        document.getElementById("captureBtn").classList.add("btn-disabled");
        document.getElementById("captureBtn").disabled = true;

        document.getElementById("startPredictCamBtn").classList.remove("btn-disabled");
        document.getElementById("startPredictCamBtn").disabled = false;
        document.getElementById("stopPredictCamBtn").classList.add("btn-disabled");
        document.getElementById("stopPredictCamBtn").disabled = true;

        if (tabName === "recordTab") {
            activeMode = "record";
            updateRecordDots();
        } else if (tabName === "trainTab") {
            activeMode = "train";
            fetchStatus();
        } else if (tabName === "predictTab") {
            activeMode = "predict";
            fetchModelStatus();
        }
    }

    // ==========================================================================
    // 1. RECORD TAB FUNCTIONALITY
    // ==========================================================================
    const recordSignSelect = document.getElementById("recordSignSelect");
    const capturedCountLabel = document.getElementById("capturedCount");
    const sampleDotsContainer = document.getElementById("sampleDotsContainer");
    const captureBtn = document.getElementById("captureBtn");

    recordSignSelect.addEventListener("change", (e) => {
        currentSignToRecord = e.target.value;
        resetRecordProgress();
    });

    document.getElementById("startRecordCamBtn").addEventListener("click", async function() {
        this.classList.add("btn-disabled");
        this.disabled = true;
        
        await initCamera(
            document.getElementById("recordVideo"),
            document.getElementById("recordCanvas"),
            document.getElementById("recordNoHandWarning"),
            document.getElementById("recordLeftBadge"),
            document.getElementById("recordRightBadge"),
            null
        );

        captureBtn.classList.remove("btn-disabled");
        captureBtn.disabled = false;
        showToast("Recording camera loaded.", "success");
    });

    function resetRecordProgress() {
        collectedSamplesCount = 0;
        updateRecordDots();
    }

    function updateRecordDots() {
        capturedCountLabel.textContent = collectedSamplesCount;
        sampleDotsContainer.innerHTML = "";
        
        for (let i = 0; i < SAMPLES_PER_SIGN; i++) {
            const dot = document.createElement("div");
            dot.className = `dot ${i < collectedSamplesCount ? 'active' : ''}`;
            dot.textContent = i + 1;
            sampleDotsContainer.appendChild(dot);
        }
    }

    async function triggerCapture() {
        if (!isWebcamRunning || activeMode !== "record") return;
        if (!isLeftHandDetected && !isRightHandDetected) {
            showToast("No hand detected. Please position hands in the camera view.", "error");
            return;
        }

        if (collectedSamplesCount >= SAMPLES_PER_SIGN) {
            showToast(`Sign '${currentSignToRecord}' already completed!`, "info");
            return;
        }

        const features = extractLandmarks(currentResults);
        
        // Visual green flash
        const flashEl = document.getElementById("recordFlash");
        flashEl.classList.add("flash-active");
        setTimeout(() => flashEl.classList.remove("flash-active"), 150);

        try {
            const response = await fetch("/api/record", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": authToken
                },
                body: JSON.stringify({
                    sign: currentSignToRecord,
                    features: features
                })
            });

            const data = await response.json();
            if (response.ok && data.success) {
                collectedSamplesCount++;
                updateRecordDots();
                showToast(`Sample ${collectedSamplesCount}/${SAMPLES_PER_SIGN} recorded!`, "success");
                
                // Update general statistics from backend
                fetchStatus();

                if (collectedSamplesCount >= SAMPLES_PER_SIGN) {
                    showToast(`✓ Sign '${currentSignToRecord}' complete!`, "success");
                }
            } else {
                showToast(data.message || "Failed to record sample.", "error");
            }
        } catch (err) {
            console.error("Capture request error:", err);
            showToast("Network error saving sample.", "error");
        }
    }

    captureBtn.addEventListener("click", triggerCapture);

    document.getElementById("skipSignBtn").addEventListener("click", () => {
        const nextIdx = recordSignSelect.selectedIndex + 1;
        if (nextIdx < recordSignSelect.options.length) {
            recordSignSelect.selectedIndex = nextIdx;
            currentSignToRecord = recordSignSelect.value;
            resetRecordProgress();
            showToast(`Skipped to sign '${currentSignToRecord}'`, "info");
        } else {
            showToast("Reached last sign.", "info");
        }
    });

    // Keyboard capture binding
    document.addEventListener("keydown", (e) => {
        if (e.code === "Space") {
            // Prevent scrolling on space press
            if (activeMode === "record" && isWebcamRunning) {
                e.preventDefault();
                triggerCapture();
            }
        } else if (e.key === "n" || e.key === "N") {
            if (activeMode === "record") {
                document.getElementById("skipSignBtn").click();
            }
        }
    });

    // ==========================================================================
    // 2. MODEL TRAINING TAB
    // ==========================================================================
    const startTrainBtn = document.getElementById("startTrainBtn");
    const trainConsole = document.getElementById("trainConsole");
    const trainStatusLabel = document.getElementById("trainStatusLabel");
    const trainProgressPercent = document.getElementById("trainProgressPercent");
    const trainProgressBarFill = document.getElementById("trainProgressBarFill");
    const statTrainLoss = document.getElementById("statTrainLoss");
    const statValAcc = document.getElementById("statValAcc");

    let isTrainingActive = false;
    let pollInterval = null;

    async function fetchStatus() {
        try {
            const response = await fetch("/api/status", {
                headers: { "Authorization": authToken }
            });
            const data = await response.json();
            if (response.ok) {
                // Update stats
                document.getElementById("statRawSamples").textContent = data.raw_samples;
                document.getElementById("statAugmentedSamples").textContent = data.augmented_samples;
                
                if (data.training_state && data.training_state.is_training) {
                    isTrainingActive = true;
                    startTrainBtn.disabled = true;
                    startTrainBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training...';
                    trainStatusLabel.textContent = `Epoch ${data.training_state.current_epoch}/${data.training_state.total_epochs}`;
                    
                    const pct = Math.round((data.training_state.current_epoch / data.training_state.total_epochs) * 100);
                    trainProgressPercent.textContent = `${pct}%`;
                    trainProgressBarFill.style.width = `${pct}%`;

                    statTrainLoss.textContent = data.training_state.last_loss.toFixed(4);
                    statValAcc.textContent = `${data.training_state.best_accuracy.toFixed(2)}%`;
                    
                    // Update log screen
                    trainConsole.textContent = data.training_state.logs.join("\n");
                    trainConsole.scrollTop = trainConsole.scrollHeight;

                    if (!pollInterval) {
                        pollInterval = setInterval(fetchStatus, 1000);
                    }
                } else {
                    if (isTrainingActive) {
                        isTrainingActive = false;
                        clearInterval(pollInterval);
                        pollInterval = null;
                        startTrainBtn.disabled = false;
                        startTrainBtn.innerHTML = '<i class="fa-solid fa-play"></i> Start Model Training';
                        trainStatusLabel.textContent = "Finished";
                        trainProgressPercent.textContent = "100%";
                        trainProgressBarFill.style.width = "100%";
                        showToast("Model training complete!", "success");
                    }
                }
            }
        } catch (err) {
            console.error("Status check failed:", err);
        }
    }

    startTrainBtn.addEventListener("click", async () => {
        const epochs = document.getElementById("trainEpochs").value;
        const batchSize = document.getElementById("trainBatchSize").value;
        const lr = document.getElementById("trainLR").value;

        trainConsole.textContent = "Starting PyTorch MLP Trainer...\nConnecting to GPU/CPU engine...";
        trainProgressBarFill.style.width = "0%";
        trainProgressPercent.textContent = "0%";
        trainStatusLabel.textContent = "Initializing";

        try {
            const response = await fetch("/api/train", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": authToken
                },
                body: JSON.stringify({
                    epochs: parseInt(epochs),
                    batch_size: parseInt(batchSize),
                    lr: parseFloat(lr)
                })
            });

            const data = await response.json();
            if (response.ok && data.success) {
                showToast("Training started in backend thread.", "info");
                isTrainingActive = true;
                startTrainBtn.disabled = true;
                startTrainBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training...';
                
                // Start polling status
                if (pollInterval) clearInterval(pollInterval);
                pollInterval = setInterval(fetchStatus, 1000);
            } else {
                showToast(data.message || "Failed to trigger training.", "error");
            }
        } catch (err) {
            console.error("Training post fail:", err);
            showToast("Failed to connect to training API.", "error");
        }
    });

    document.getElementById("clearLogsBtn").addEventListener("click", () => {
        trainConsole.textContent = "";
    });

    // ==========================================================================
    // 3. PREDICT TAB FUNCTIONALITY
    // ==========================================================================
    const startPredictCamBtn = document.getElementById("startPredictCamBtn");
    const stopPredictCamBtn = document.getElementById("stopPredictCamBtn");
    const modelStatusText = document.getElementById("modelStatusText");
    const modelStatusColor = document.getElementById("modelStatusColor");
    
    const predictionResultOverlay = document.getElementById("predictionResultOverlay");
    const predictedSignVal = document.getElementById("predictedSignVal");
    const predictedConfidenceVal = document.getElementById("predictedConfidenceVal");
    const predictDistributionBars = document.getElementById("predictDistributionBars");

    let isPredicting = false;
    let predictTimeoutId = null;
    let knownClasses = []; // Populate dynamically from labels pkl

    async function fetchModelStatus() {
        try {
            const response = await fetch("/api/status", {
                headers: { "Authorization": authToken }
            });
            const data = await response.json();
            if (response.ok) {
                if (data.model_trained) {
                    modelStatusText.textContent = "Model Loaded & Ready";
                    modelStatusColor.style.color = "var(--success)";
                    knownClasses = data.classes || [];
                    setupChartBars(knownClasses);
                } else {
                    modelStatusText.textContent = "No trained model found. Run training first.";
                    modelStatusColor.style.color = "var(--error)";
                    predictionResultOverlay.style.display = "none";
                }
            }
        } catch (err) {
            console.error("Model status check failed:", err);
        }
    }

    function setupChartBars(classesList) {
        if (!classesList || classesList.length === 0) return;
        predictDistributionBars.innerHTML = "";
        
        classesList.forEach(cls => {
            const barGroup = document.createElement("div");
            barGroup.className = "chart-bar-group";
            barGroup.innerHTML = `
                <div class="chart-bar" id="bar-${cls}" style="height: 0px;">
                    <span class="chart-bar-val" id="bar-val-${cls}">0%</span>
                </div>
                <span class="chart-bar-label">Sign ${cls}</span>
            `;
            predictDistributionBars.appendChild(barGroup);
        });
    }

    startPredictCamBtn.addEventListener("click", async function() {
        this.classList.add("btn-disabled");
        this.disabled = true;

        await initCamera(
            document.getElementById("predictVideo"),
            document.getElementById("predictCanvas"),
            document.getElementById("predictNoHandWarning"),
            document.getElementById("predictLeftBadge"),
            document.getElementById("predictRightBadge"),
            handlePredictionFrame
        );

        stopPredictCamBtn.classList.remove("btn-disabled");
        stopPredictCamBtn.disabled = false;
        isPredicting = true;
        predictionResultOverlay.style.display = "flex";
        showToast("Predictions camera started.", "success");
    });

    stopPredictCamBtn.addEventListener("click", function() {
        this.classList.add("btn-disabled");
        this.disabled = true;

        stopCamera(document.getElementById("predictVideo"));
        
        startPredictCamBtn.classList.remove("btn-disabled");
        startPredictCamBtn.disabled = false;
        isPredicting = false;
        predictionResultOverlay.style.display = "none";
        
        // Reset bars
        knownClasses.forEach(cls => {
            const bar = document.getElementById(`bar-${cls}`);
            const barVal = document.getElementById(`bar-val-${cls}`);
            if (bar && barVal) {
                bar.style.height = "0px";
                barVal.textContent = "0%";
            }
        });

        showToast("Predictions stopped.", "info");
    });

    let lastPredictTime = 0;
    const INFERENCE_THROTTLE_MS = 150; // Throttle requests to ~6 FPS to save CPU/Network

    async function handlePredictionFrame(results) {
        if (!isPredicting) return;
        
        const now = Date.now();
        if (now - lastPredictTime < INFERENCE_THROTTLE_MS) return;
        
        // Check if any hand is visible
        if (!isLeftHandDetected && !isRightHandDetected) {
            predictedSignVal.textContent = "—";
            predictedConfidenceVal.textContent = "0%";
            return;
        }

        lastPredictTime = now;
        const features = extractLandmarks(results);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": authToken
                },
                body: JSON.stringify({ features: features })
            });

            const data = await response.json();
            if (response.ok && data.success) {
                // Update Prediction overlays
                predictedSignVal.textContent = `Sign ${data.predicted_class}`;
                predictedConfidenceVal.textContent = `${Math.round(data.confidence * 100)}%`;

                // Update charts/bars
                if (data.probabilities && knownClasses.length > 0) {
                    knownClasses.forEach(cls => {
                        const prob = data.probabilities[cls] || 0.0;
                        const pct = Math.round(prob * 100);
                        const bar = document.getElementById(`bar-${cls}`);
                        const barVal = document.getElementById(`bar-val-${cls}`);
                        
                        if (bar && barVal) {
                            bar.style.height = `${pct * 1.5}px`; // scale for rendering
                            barVal.textContent = `${pct}%`;
                        }
                    });
                }
            }
        } catch (err) {
            console.error("Predict server request failed:", err);
        }
    }

    // Initialize stats
    fetchStatus();
});
