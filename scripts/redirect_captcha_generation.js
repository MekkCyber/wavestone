if (performance.navigation.type === 1 && window.location.search.includes('captchaType=Python')) {
    // Redirect to the "/auth/generateCaptcha" page
    window.location.href = '/auth/generateCaptcha';
}