function validateInput_MNIST(input) {
    input.value = input.value.replace(/\D/g, '');
    if (input.value.trim() === '') {
        input.setCustomValidity('Please enter a 4-digit number');
        enableSubmitButton(false);
    } else if (/^\d{4}$/.test(input.value)) {
    input.setCustomValidity('');
    enableSubmitButton(true);
    } else {
    input.setCustomValidity('Please enter a 4-digit number');
    enableSubmitButton(false);
    } 
}
function validateInput_EMNIST(input) {
    // Remove any non-alphanumeric characters from the input value
    input.value = input.value.replace(/[^0-9a-zA-Z]/g, '');
    
    if (input.value.trim() === '') {
        input.setCustomValidity('Please enter a 4-character string containing digits and/or letters (upper or lower case)');
        enableSubmitButton(false);
    } else if (/^[0-9a-zA-Z]{4}$/.test(input.value)) {
        input.setCustomValidity('');
        enableSubmitButton(true);
    } else {
        input.setCustomValidity('Please enter a 4-character string containing digits and/or letters (upper or lower case)');
        enableSubmitButton(false);
    } 
}
function enableSubmitButton(enable) {
    var submitButton = document.querySelector('button[type="submit"]');
    submitButton.disabled = !enable;
}