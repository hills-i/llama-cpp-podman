// This script fetches available models from the API and populates the model select box
window.addEventListener('DOMContentLoaded', async () => {
    const modelSelect = document.getElementById('model');
    try {
        const response = await fetch('https://localhost:8443/v1/models');
        if (!response.ok) throw new Error('Failed to fetch models from API');
        const data = await response.json();
        if (data && data.data && Array.isArray(data.data)) {
            modelSelect.innerHTML = '';
            data.data.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.id;
                modelSelect.appendChild(option);
            });
        }
    } catch (e) {
        // fallback: show static option if API fails
        if (!modelSelect.querySelector('option')) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No models available';
            modelSelect.appendChild(option);
        }
    }
});
