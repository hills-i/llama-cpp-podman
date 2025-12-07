// ui.js - Shared UI logic for email tools

// Detect dark mode
function applyDarkMode() {
    if (!document.body) return; // Prevent error if body is not loaded
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.body.classList.add('dark-mode');
    } else {
        document.body.classList.remove('dark-mode');
    }
}

// Run after DOMContentLoaded to ensure document.body exists
document.addEventListener('DOMContentLoaded', () => {
    applyDarkMode();
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyDarkMode);
});

// Shared UI event setup
function setupCommonUI({
    submitBtn,
    responseDiv,
    loadingDiv,
    copyBtn,
    clearBtn,
    promptInput
}) {
    // Copy button handler
    copyBtn.addEventListener('click', () => {
        const text = responseDiv.textContent;  // Use textContent to prevent XSS
        navigator.clipboard.writeText(text)
            .then(() => {
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                setTimeout(() => {
                    copyBtn.textContent = originalText;
                }, 2000);
            })
            .catch(err => {
                console.error('Failed to copy text:', err);
            });
    });

    // Clear button handler
    clearBtn.addEventListener('click', () => {
        responseDiv.innerHTML = '';
        copyBtn.classList.add('hidden');
    });

    // Create and add think tag toggle button
    const toggleThinkBtn = document.createElement('button');
    toggleThinkBtn.textContent = 'Toggle think display';
    toggleThinkBtn.className = 'btn-small';
    toggleThinkBtn.style.marginLeft = '8px';

    // Add to btn-group
    const btnGroup = document.querySelector('.btn-group');
    btnGroup.appendChild(toggleThinkBtn);

    // Add style for hiding think tag
    const style = document.createElement('style');
    style.textContent = `
        #response think {
            display: none;
            white-space: pre-wrap;
            background-color: #f0f0f0;
            border-left: 3px solid #5D5CDE;
            padding-left: 8px;
            margin: 4px 0;
            font-style: italic;
            color: #555;
        }
        #response.show-think think {
            display: block;
        }
    `;
    document.head.appendChild(style);

    // Toggle display event
    toggleThinkBtn.addEventListener('click', () => {
        responseDiv.classList.toggle('show-think');
    });

    // Submit on Ctrl+Enter
    if (promptInput) {
        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                submitBtn.click();
            }
        });
    }
}

// Shared submit handler for both UIs
function handleSubmit({
    getPrompt,
    modelSelect,
    responseDiv,
    loadingDiv,
    copyBtn,
    submitBtn,
    errorMsg = 'API error occurred'
}) {
    return async () => {
        const model = modelSelect.value;
        const prompt = getPrompt();

        if (!prompt) {
            alert('Please enter your prompt.');
            return;
        }

        loadingDiv.classList.remove('hidden');
        responseDiv.innerHTML = '';
        responseDiv.classList.remove('error-text');
        copyBtn.classList.add('hidden');
        submitBtn.disabled = true;

        try {
            const requestBody = {
                model: model,
                messages: [
                    { role: 'user', content: prompt }
                ],
                max_tokens: 2000,
                stream: false
            };
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error?.message || errorMsg);
            }

            let result = data.choices[0].message.content;
            result = result.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            result = result.replace(/```([\s\S]*?)```/g, (_, code) => `<blockquote>${code}</blockquote>`);
            responseDiv.innerHTML = result;
            copyBtn.classList.remove('hidden');
        } catch (error) {
            responseDiv.innerHTML = `Error: ${error.message}`;
            responseDiv.classList.add('error-text');
        } finally {
            loadingDiv.classList.add('hidden');
            submitBtn.disabled = false;
        }
    };
}

function initProofreadUI(prompt_template_1, prompt_template_2) {
    document.addEventListener('DOMContentLoaded', () => {
        const modelSelect = document.getElementById('model');
        const promptInput = document.getElementById('prompt');
        const submitBtn = document.getElementById('submitBtn');
        const responseDiv = document.getElementById('response');
        const loadingDiv = document.getElementById('loading');
        const copyBtn = document.getElementById('copyBtn');
        const clearBtn = document.getElementById('clearBtn');

        setupCommonUI({ submitBtn, responseDiv, loadingDiv, copyBtn, clearBtn, promptInput });

        submitBtn.addEventListener('click', handleSubmit({
            getPrompt: () => prompt_template_1 + promptInput.value.trim() + prompt_template_2,
            modelSelect,
            responseDiv,
            loadingDiv,
            copyBtn,
            submitBtn,
            errorMsg: 'API error occurred'
        }));
    });
}

function initReplyLocalUI(prompt_template_1, prompt_template_2, prompt_template_3) {
    document.addEventListener('DOMContentLoaded', () => {
        const modelSelect = document.getElementById('model');
        const emailInput = document.getElementById('email');
        const promptInput = document.getElementById('prompt');
        const submitBtn = document.getElementById('submitBtn');
        const responseDiv = document.getElementById('response');
        const loadingDiv = document.getElementById('loading');
        const copyBtn = document.getElementById('copyBtn');
        const clearBtn = document.getElementById('clearBtn');

        setupCommonUI({ submitBtn, responseDiv, loadingDiv, copyBtn, clearBtn, promptInput });

        submitBtn.addEventListener('click', handleSubmit({
            getPrompt: () => prompt_template_1 + emailInput.value.trim() + prompt_template_2 + promptInput.value.trim() + prompt_template_3,
            modelSelect,
            responseDiv,
            loadingDiv,
            copyBtn,
            submitBtn,
            errorMsg: 'Enter the reply content'
        }));
    });
}

function initTranslateUI(prompt_ja_to_en, prompt_en_to_ja, prompt_suffix) {
    document.addEventListener('DOMContentLoaded', () => {
        const modelSelect = document.getElementById('model');
        const promptInput = document.getElementById('prompt');
        const submitBtn = document.getElementById('submitBtn');
        const responseDiv = document.getElementById('response');
        const loadingDiv = document.getElementById('loading');
        const copyBtn = document.getElementById('copyBtn');
        const clearBtn = document.getElementById('clearBtn');

        setupCommonUI({ submitBtn, responseDiv, loadingDiv, copyBtn, clearBtn, promptInput });

        submitBtn.addEventListener('click', handleSubmit({
            getPrompt: () => {
                const direction = document.querySelector('input[name="direction"]:checked').value;
                const prefix = direction === 'ja-en' ? prompt_ja_to_en : prompt_en_to_ja;
                return prefix + promptInput.value.trim() + prompt_suffix;
            },
            modelSelect,
            responseDiv,
            loadingDiv,
            copyBtn,
            submitBtn,
            errorMsg: 'Translation error occurred'
        }));
    });
}

// Load side menu from external JSON file and insert as links into .sidemenu
function loadSideMenu(menuPath = 'json/sidemenu.json') {
    document.addEventListener('DOMContentLoaded', () => {
        const sidemenu = document.querySelector('.sidemenu');
        if (!sidemenu) return;
        fetch(menuPath)
            .then(res => res.json())
            .then(links => {
                sidemenu.innerHTML = '<nav class="sidemenu">' +
                    links.map(link =>
                        `<a href="${link.href}">${link.label}</a>`
                    ).join('') +
                    '</nav>';
            })
            .catch(() => {
                sidemenu.innerHTML = 'Menu failed to load';
            });
    });
}
