// login.js
document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.getElementById("loginForm");
    const alertMessage = document.getElementById("alertMessage");
    const alertText = document.getElementById("alertText");
    const submitBtn = document.getElementById("submitBtn");

    checkSession();

    loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const username = document.getElementById("username").value.trim();
        const password = document.getElementById("password").value;

        submitBtn.disabled = true;
        const origText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Signing in...';
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
                localStorage.setItem("authToken", data.token);
                const user = data.user;
                localStorage.setItem("username", user.username);
                localStorage.setItem("fullname", user.fullname);
                localStorage.setItem("role", user.role);
                document.cookie = `authToken=${data.token}; path=/`;
                window.location.href = user.role === "admin" ? "admin.html" : "index.html";
            } else {
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

    async function checkSession() {
        const authToken = localStorage.getItem("authToken");
        if (!authToken) return;

        try {
            const response = await fetch("/api/session", {
                headers: { "Authorization": `Bearer ${authToken}` }
            });
            if (response.ok) {
                const user = (await response.json()).user;
                localStorage.setItem("role", user.role);
                localStorage.setItem("username", user.username);
                window.location.href = user.role === "admin" ? "admin.html" : "index.html";
                return;
            }
        } catch (err) {
            console.error("Session validation failed:", err);
        }

        localStorage.removeItem("authToken");
        localStorage.removeItem("username");
        localStorage.removeItem("fullname");
        localStorage.removeItem("role");
    }

    function showError(message) {
        alertText.textContent = message;
        alertMessage.style.display = "flex";
        alertMessage.classList.add("shake-animation");
        setTimeout(() => alertMessage.classList.remove("shake-animation"), 500);
    }
});
