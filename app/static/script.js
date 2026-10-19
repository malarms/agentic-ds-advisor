document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('analysisForm');
    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const errorMessage = document.getElementById('errorMessage');
    const successMessage = document.getElementById('successMessage');
    const htmlLink = document.getElementById('htmlLink');
    const jsonLink = document.getElementById('jsonLink');
    const resultModality = document.getElementById('resultModality');
    const resultFindings = document.getElementById('resultFindings');
    const resultRecommendations = document.getElementById('resultRecommendations');
    const resultStatus = document.getElementById('resultStatus');
    const errorMessageEl = document.getElementById('errorMessage');
    const successMessageEl = document.getElementById('successMessage');
    const htmlLinkEl = document.getElementById('htmlLink');
    const jsonLinkEl = document.getElementById('jsonLink');

    // Client-side validation
    function validateForm() {
        const file = document.getElementById('datasetFile').files[0];
        const request = document.getElementById('request').value.trim();

        if (!file) {
            showError('Please select a dataset file.');
            return false;
        }

        if (!request) {
            showError('Please enter an analysis objective.');
            return false;
        }

        // Check file size (100MB)
        if (file.size > 100 * 1024 * 1024) {
            showError('File size exceeds 100MB limit.');
            return false;
        }

        // Basic extension validation
        const allowedExtensions = [
            '.csv', '.xlsx', '.xls', '.parquet',
            '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp',
            '.wav', '.flac', '.ogg', '.aiff', '.aif',
            '.zip'
        ];
        const ext = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        if (!allowedExtensions.includes(ext)) {
            showError(`Unsupported file type: ${ext}. Supported: CSV, Excel, Parquet, PNG, JPEG, BMP, TIFF, WebP, WAV, FLAC, OGG, AIFF, ZIP.`);
            return false;
        }

        return true;
    }

    function showError(message) {
        errorMessageEl.textContent = message;
        errorMessageEl.classList.remove('hidden');
        successMessageEl.classList.add('hidden');
    }

    function showSuccess(message) {
        successMessageEl.textContent = message;
        successMessageEl.classList.remove('hidden');
        errorMessageEl.classList.add('hidden');
    }

    function hideMessages() {
        errorMessageEl.classList.add('hidden');
        successMessageEl.classList.add('hidden');
    }

    function setLoading(isLoading) {
        submitBtn.disabled = isLoading;
        if (isLoading) {
            submitBtn.textContent = 'Analyzing...';
            document.getElementById('analysisForm').classList.add('hidden');
            loading.classList.remove('hidden');
            results.classList.add('hidden');
        } else {
            submitBtn.textContent = 'Run Analysis';
            document.getElementById('analysisForm').classList.remove('hidden');
            loading.classList.add('hidden');
        }
    }

    async function handleSubmit(event) {
        event.preventDefault();

        if (!validateForm()) {
            return;
        }

        hideMessages();
        setLoading(true);

        const formData = new FormData();
        const file = document.getElementById('datasetFile').files[0];
        const request = document.getElementById('request').value.trim();
        const modality = document.getElementById('modality').value;

        formData.append('file', file);
        formData.append('request', request);
        if (modality) {
            formData.append('modality', modality);
        }

        try {
            const response = await fetch('/advisor', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                const errorDetail = data.detail || `Request failed with status ${response.status}`;
                throw new Error(errorDetail);
            }

            // Display results
            displayResults(data);
            showSuccess('Analysis completed successfully!');

        } catch (error) {
            showError(error.message || 'An unexpected error occurred.');
        } finally {
            setLoading(false);
        }
    }

    function displayResults(data) {
        // Update summary metrics
        document.getElementById('resultModality').textContent = data.modality || '—';
        document.getElementById('resultFindings').textContent = data.summary?.findings ?? '—';
        document.getElementById('resultRecommendations').textContent = data.summary?.recommendations ?? '—';
        document.getElementById('resultStatus').textContent = data.summary?.advisory_status ?? '—';

        // Set download links
        if (data.download) {
            htmlLink.href = data.download.html;
            jsonLink.href = data.download.json;
        }

        // Show results section
        document.getElementById('results').classList.remove('hidden');

        // Scroll to results
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Event listeners
    document.getElementById('analysisForm').addEventListener('submit', handleSubmit);

    // Allow starting a new analysis
    document.getElementById('analysisForm').addEventListener('reset', function() {
        document.getElementById('results').classList.add('hidden');
        hideMessages();
    });

    // Clear error when user interacts with form
    document.getElementById('datasetFile').addEventListener('change', hideMessages);
    document.getElementById('request').addEventListener('input', hideMessages);
    document.getElementById('modality').addEventListener('change', hideMessages);
});