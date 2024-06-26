document.addEventListener('DOMContentLoaded', function() {
    // Get the selected captcha type from the query parameters in the URL
    const urlParams = new URLSearchParams(window.location.search);
    const captchaType = urlParams.get('captchaType');

    // Set the selected option in the captcha selector
    const captchaSelector = document.getElementById('captchaMethod');
    if (captchaSelector) {
        // Iterate over options and set the selected attribute for the corresponding captcha type
        Array.from(captchaSelector.options).forEach(option => {
            if (option.value === captchaType) {
                option.selected = true;
            }
        });
    }

    // Add event listener to reload the page when captcha selector changes
    captchaSelector.addEventListener('change', function() {
        const selectedCaptchaType = this.value;
        if (selectedCaptchaType === 'Python') {
            //window.location.href = '/auth/generateCaptcha';
            window.location.href = '/auth/login?captchaType=Python';
            // .then(response => response.text())
            // .then(message => console.log(message))
            // .catch(error => console.error(error))
            // .then(() => {
            //     // Redirect after code completion
            //     window.location.href = '/auth/login?captchaType=Python';
            // })
        } else {
            // Redirect without executing Python code
            console.log("Changing captcha type to :", selectedCaptchaType)
            window.location.href = '/auth/login?captchaType=' + selectedCaptchaType;
        }
    });

    // Get all other selectors
    const selectors = document.querySelectorAll('.selector-container select:not(#captchaMethod)');

    // Add event listener to reload the page when any other selector changes
    selectors.forEach(selector => {
        selector.addEventListener('change', function() {
            // No need to capture captchaType here
            window.location.reload();
        });
    });
});