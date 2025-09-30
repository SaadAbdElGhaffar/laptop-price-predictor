// JavaScript for Laptop Price Predictor

// API endpoint
const API_URL = 'http://127.0.0.1:5000/predict';

// Form validation rules
const validationRules = {
    Ram: { min: 2, max: 64 },
    Weight: { min: 0.5, max: 5.0 },
    Ppi: { min: 100, max: 400 },
    HDD: { min: 0, max: 2000 },
    SSD: { min: 0, max: 2000 }
};

// Main prediction function
async function predictPrice() {
    // Show loading state
    showLoading();
    
    // Validate form
    if (!validateForm()) {
        hideLoading();
        return;
    }
    
    // Collect form data
    const data = collectFormData();
    
    try {
        // Make API request
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        // Display result
        if (response.ok && result.price) {
            showSuccess(`Predicted Price: $${result.price}`);
        } else {
            showError(`Error: ${result.error || 'Unknown error occurred'}`);
        }
        
    } catch (error) {
        showError(`Network Error: ${error.message}`);
        console.error('Prediction error:', error);
    } finally {
        hideLoading();
    }
}

// Collect all form data
function collectFormData() {
    const data = {};
    const formElements = [
        'Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen',
        'Ips', 'Ppi', 'Cpu', 'HDD', 'SSD', 'Gpu', 'OpSys'
    ];
    
    formElements.forEach(elementId => {
        const element = document.getElementById(elementId);
        let value = element.value;
        
        // Convert numeric fields
        if (['Ram', 'Weight', 'Touchscreen', 'Ips', 'Ppi', 'HDD', 'SSD'].includes(elementId)) {
            value = parseFloat(value);
        }
        
        data[elementId] = value;
    });
    
    return data;
}

// Form validation
function validateForm() {
    let isValid = true;
    
    // Clear previous validation states
    clearValidationStates();
    
    // Validate each field with rules
    Object.keys(validationRules).forEach(fieldName => {
        const field = document.getElementById(fieldName);
        const value = parseFloat(field.value);
        const rules = validationRules[fieldName];
        
        if (value < rules.min || value > rules.max) {
            markFieldError(fieldName);
            isValid = false;
        } else {
            markFieldSuccess(fieldName);
        }
    });
    
    // Validate required fields
    const requiredFields = ['Company', 'TypeName', 'Cpu', 'Gpu', 'OpSys'];
    requiredFields.forEach(fieldName => {
        const field = document.getElementById(fieldName);
        if (!field.value) {
            markFieldError(fieldName);
            isValid = false;
        } else {
            markFieldSuccess(fieldName);
        }
    });
    
    if (!isValid) {
        showError('Please correct the highlighted fields');
    }
    
    return isValid;
}

// Mark field as error
function markFieldError(fieldName) {
    const formGroup = document.getElementById(fieldName).closest('.form-group');
    formGroup.classList.remove('success');
    formGroup.classList.add('error');
}

// Mark field as success
function markFieldSuccess(fieldName) {
    const formGroup = document.getElementById(fieldName).closest('.form-group');
    formGroup.classList.remove('error');
    formGroup.classList.add('success');
}

// Clear validation states
function clearValidationStates() {
    document.querySelectorAll('.form-group').forEach(group => {
        group.classList.remove('error', 'success');
    });
}

// Show loading state
function showLoading() {
    const resultDiv = document.getElementById('result');
    resultDiv.className = 'text-center';
    resultDiv.innerHTML = '<div class="loading"></div>Predicting price...';
}

// Hide loading state
function hideLoading() {
    // Loading will be hidden when result is shown
}

// Show success message
function showSuccess(message) {
    const resultDiv = document.getElementById('result');
    resultDiv.className = 'text-center success';
    resultDiv.innerHTML = message;
}

// Show error message
function showError(message) {
    const resultDiv = document.getElementById('result');
    resultDiv.className = 'text-center error';
    resultDiv.innerHTML = message;
}

// Reset form to default values
function resetForm() {
    document.getElementById('Company').value = 'Dell';
    document.getElementById('TypeName').value = 'Notebook';
    document.getElementById('Ram').value = '8';
    document.getElementById('Weight').value = '2.0';
    document.getElementById('Touchscreen').value = '0';
    document.getElementById('Ips').value = '0';
    document.getElementById('Ppi').value = '141.21';
    document.getElementById('Cpu').value = 'Intel Core i5';
    document.getElementById('HDD').value = '0';
    document.getElementById('SSD').value = '256';
    document.getElementById('Gpu').value = 'Intel';
    document.getElementById('OpSys').value = 'Windows';
    
    clearValidationStates();
    document.getElementById('result').innerHTML = '';
    document.getElementById('result').className = 'text-center';
}

// Load sample configurations
function loadSampleConfig(configName) {
    const configs = {
        budget: {
            Company: 'Acer',
            TypeName: 'Notebook',
            Ram: 4,
            Weight: 2.2,
            Touchscreen: 0,
            Ips: 0,
            Ppi: 141.21,
            Cpu: 'Intel Core i3',
            HDD: 500,
            SSD: 0,
            Gpu: 'Intel',
            OpSys: 'Windows'
        },
        gaming: {
            Company: 'MSI',
            TypeName: 'Gaming',
            Ram: 16,
            Weight: 2.8,
            Touchscreen: 0,
            Ips: 1,
            Ppi: 180,
            Cpu: 'Intel Core i7',
            HDD: 0,
            SSD: 512,
            Gpu: 'Nvidia',
            OpSys: 'Windows'
        },
        premium: {
            Company: 'Apple',
            TypeName: 'Ultrabook',
            Ram: 16,
            Weight: 1.4,
            Touchscreen: 0,
            Ips: 1,
            Ppi: 220,
            Cpu: 'Intel Core i7',
            HDD: 0,
            SSD: 512,
            Gpu: 'Intel',
            OpSys: 'macOS'
        }
    };
    
    const config = configs[configName];
    if (config) {
        Object.keys(config).forEach(key => {
            document.getElementById(key).value = config[key];
        });
        clearValidationStates();
    }
}

// Initialize the application
function initializeApp() {
    // Add event listeners for real-time validation
    Object.keys(validationRules).forEach(fieldName => {
        const field = document.getElementById(fieldName);
        field.addEventListener('blur', () => validateField(fieldName));
        field.addEventListener('input', () => clearFieldValidation(fieldName));
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            predictPrice();
        }
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            resetForm();
        }
    });
    
    console.log('Laptop Price Predictor initialized');
}

// Validate individual field
function validateField(fieldName) {
    const field = document.getElementById(fieldName);
    const value = parseFloat(field.value);
    const rules = validationRules[fieldName];
    
    if (value < rules.min || value > rules.max) {
        markFieldError(fieldName);
        return false;
    } else {
        markFieldSuccess(fieldName);
        return true;
    }
}

// Clear field validation
function clearFieldValidation(fieldName) {
    const formGroup = document.getElementById(fieldName).closest('.form-group');
    formGroup.classList.remove('error', 'success');
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeApp);