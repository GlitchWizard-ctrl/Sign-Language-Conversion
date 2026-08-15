document.addEventListener("DOMContentLoaded", () => {
    const registerForm      = document.getElementById("registerForm");
    const alertMessage      = document.getElementById("alertMessage");
    const alertText         = document.getElementById("alertText");
    const successMessage    = document.getElementById("successMessage");
    const submitBtn         = document.getElementById("submitBtn");
    const passwordInput     = document.getElementById("password");
    const confirmInput      = document.getElementById("confirmPassword");
    const passwordHint      = document.getElementById("passwordHint");

    // Live feedback while typing, so users aren't surprised at submit time
    function updatePasswordHint() {
        const pwd = passwordInput.value;
        const confirm = confirmInput.value;

        if (!pwd && !confirm) {
            passwordHint.className = "capture-status";
            passwordHint.innerHTML = '<i class="fa-solid fa-circle-info"></i> Use at least 6 characters.';
            return;
        }
        if (pwd.length > 0 && pwd.length < 6) {
            passwordHint.className = "capture-status error";
            passwordHint.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Password needs at least 6 characters.';
            return;
        }
        if (confirm.length > 0 && pwd !== confirm) {
            passwordHint.className = "capture-status error";
            passwordHint.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Passwords do not match.';
            return;
        }
        if (pwd.length >= 6 && confirm.length >= 6 && pwd === confirm) {
            passwordHint.className = "capture-status ok";
            passwordHint.innerHTML = '<i class="fa-solid fa-circle-check"></i> Looks good.';
            return;
        }
        passwordHint.className = "capture-status";
        passwordHint.innerHTML = '<i class="fa-solid fa-circle-info"></i> Use at least 6 characters.';
    }

    passwordInput.addEventListener("input", updatePasswordHint);
    confirmInput.addEventListener("input", updatePasswordHint);

    registerForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const fullname = document.getElementById("fullname").value.trim();
        const email = document.getElementById("email").value.trim();
        const username = document.getElementById("username").value.trim();
        const password = passwordInput.value;
        const confirmPassword = confirmInput.value;

        if (username.length < 3) {
            return showError("Username must be at least 3 characters.");
        }
        if (password.length < 6) {
            return showError("Password must be at least 6 characters.");
        }
        if (password !== confirmPassword) {
            return showError("Passwords do not match.");
        }

        submitBtn.disabled = true;
        const origHTML = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> <span>Creating account...</span>';
        alertMessage.style.display = "none";
        successMessage.style.display = "none";

        try {
            const response = await fetch("/api/register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fullname, email, username, password })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                successMessage.style.display = "flex";
                registerForm.reset();
                passwordHint.className = "capture-status";
                passwordHint.innerHTML = '<i class="fa-solid fa-circle-info"></i> Use at least 6 characters.';
                submitBtn.innerHTML = '<i class="fa-solid fa-circle-check"></i> <span>Account created</span>';
                setTimeout(() => {
                    window.location.href = "login.html";
                }, 1800);
                return; // keep button in success state until redirect
            } else {
                showError(data.message || "Registration failed. Please try again.");
            }
        } catch (err) {
            console.error("Registration request failed:", err);
            showError("Server unreachable. Please make sure the backend is running.");
        }

        submitBtn.disabled = false;
        submitBtn.innerHTML = origHTML;
    });

    function showError(message) {
        alertText.textContent = message;
        alertMessage.style.display = "flex";
        alertMessage.classList.add("shake-animation");
        setTimeout(() => alertMessage.classList.remove("shake-animation"), 400);
    }
});