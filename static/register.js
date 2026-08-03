document.addEventListener("DOMContentLoaded", () => {
    const registerForm = document.getElementById("registerForm");
    const alertMessage = document.getElementById("alertMessage");
    const alertText = document.getElementById("alertText");
    const successMessage = document.getElementById("successMessage");
    const submitBtn = document.getElementById("submitBtn");

    registerForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const fullname = document.getElementById("fullname").value.trim();
        const email = document.getElementById("email").value.trim();
        const username = document.getElementById("username").value.trim();
        const password = document.getElementById("password").value;
        const confirmPassword = document.getElementById("confirmPassword").value;

        if (password.length < 6) {
            return showError("Password must be at least 6 characters.");
        }
        if (password !== confirmPassword) {
            return showError("Passwords do not match.");
        }

        submitBtn.disabled = true;
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Creating account...';
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
                setTimeout(() => window.location.href = "login.html", 1800);
            } else {
                showError(data.message || "Unable to create account.");
            }
        } catch (err) {
            console.error("Registration failed:", err);
            showError("Server unreachable. Please ensure backend is running.");
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    });

    function showError(message) {
        alertText.textContent = message;
        alertMessage.style.display = "flex";
        alertMessage.classList.add("shake-animation");
        setTimeout(() => alertMessage.classList.remove("shake-animation"), 500);
    }
});
