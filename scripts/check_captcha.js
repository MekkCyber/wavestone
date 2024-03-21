document.addEventListener('DOMContentLoaded', function () {
    // Reset the value inside the input field
    var captchaInput = document.getElementById('captcha_input');
    if (captchaInput) {
      captchaInput.value = '';
    }

    var urlParams = new URLSearchParams(window.location.search);
    var incorrectCaptcha = urlParams.get('incorrectCaptcha');

    if (incorrectCaptcha === 'true') {
    // Inform the user about the incorrect captcha
        captchaInput.setCustomValidity("Incorrect Captcha");
    }

    // Disable the submit button on page load
    enableSubmitButton(false);
});
function validateCaptcha() {
    return true
    var captchaInput = document.getElementById('captcha_input');
    var captchaValue = <%= captcha_value %>; // Get the server-side constant value

    // Check if the input value is equal to the constant captcha_value
    if (captchaInput.value.trim() === captchaValue.toString()) {
        console.log("bon");
        captchaInput.setCustomValidity('');
        enableSubmitButton(true);
        return true; // Allow form submission
    } else {
        console.log("Mauvais")
        enableSubmitButton(false);
        window.location.reload();
        var url = window.location.href + '?incorrectCaptcha=true';
        window.location.href = url;
        return false; // Prevent form submission
    }
}
