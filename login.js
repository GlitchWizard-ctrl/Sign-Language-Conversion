document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.getElementById("loginForm");
    const alertMessage = document.getElementById("alertMessage");
    const alertText = document.getElementById("alertText");
    const submitBtn = document.getElementById("submitBtn");

    loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const username = document.getElementById("username").value.trim();
        const password = document.getElementById("password").value;

        // Visual feedback
        submitBtn.disabled = true;
        const origText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Authenticating...';
        alertMessage.style.display = "none";

        try {
            const response = await fetch("/api/login", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ username, password })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // Save session token and redirect
                localStorage.setItem("authToken", data.token);
                localStorage.setItem("username", username);
                window.location.href = "index.html";
            } else {
                // Show API error
                showError(data.message || "Invalid credentials. Please try again.");
            }
        } catch (err) {
            console.error("Login request failed:", err);
            showError("Server unreachable. Please make sure backend is running.");
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = origText;
        }
    });

    function showError(message) {
        alertText.textContent = message;
        alertMessage.style.display = "flex";
        alertMessage.classList.add("shake-animation");
        setTimeout(() => alertMessage.classList.remove("shake-animation"), 500);
    }
});
