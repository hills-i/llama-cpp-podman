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
    let activeController = null;
    const originalSubmitText = submitBtn.textContent;

    return async () => {
        if (activeController) {
            activeController.abort();
            return;
        }

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
        submitBtn.disabled = false;
        submitBtn.textContent = 'Stop';
        submitBtn.classList.add('btn-stop');

        const controller = new AbortController();
        activeController = controller;
        let rawText = '';
        let aborted = false;

        const renderResponse = () => {
            // Escape first, then apply a tiny, safe subset of formatting.
            // This ensures untrusted model output never becomes executable HTML.
            responseDiv.innerHTML = formatSafeResponseHtml(escapeHtml(rawText));
        };

        try {
            const requestBody = {
                model: model,
                messages: [
                    { role: 'user', content: prompt }
                ],
                max_tokens: 2000,
                stream: true
            };
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody),
                signal: controller.signal
            });

            if (!response.ok) {
                throw new Error(await getApiErrorMessage(response, errorMsg));
            }

            if (!response.body) {
                throw new Error('Streaming response is not available in this browser.');
            }

            await readChatCompletionStream(response.body, (contentDelta) => {
                rawText += contentDelta;
                renderResponse();
            });
        } catch (error) {
            aborted = error.name === 'AbortError';

            if (!aborted) {
                responseDiv.textContent = `Error: ${error.message}`;
                responseDiv.classList.add('error-text');
                rawText = '';
            }
        } finally {
            activeController = null;
            loadingDiv.classList.add('hidden');
            submitBtn.textContent = originalSubmitText;
            submitBtn.classList.remove('btn-stop');

            if (rawText) {
                renderResponse();
                copyBtn.classList.remove('hidden');
            }
        }
    };
}

async function readChatCompletionStream(body, onContentDelta) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() ?? '';

        for (const line of lines) {
            if (processChatCompletionStreamLine(line, onContentDelta)) {
                return;
            }
        }
    }

    buffer += decoder.decode();

    if (buffer.trim()) {
        const lines = buffer.split(/\r?\n/);
        for (const line of lines) {
            if (processChatCompletionStreamLine(line, onContentDelta)) {
                return;
            }
        }
    }
}

function processChatCompletionStreamLine(line, onContentDelta) {
    const trimmed = line.trim();

    if (!trimmed || trimmed.startsWith(':')) {
        return false;
    }

    if (!trimmed.startsWith('data:')) {
        return false;
    }

    const data = trimmed.slice(5).trim();

    if (data === '[DONE]') {
        return true;
    }

    const parsed = JSON.parse(data);
    const contentDelta = parsed?.choices?.[0]?.delta?.content ?? '';

    if (contentDelta) {
        onContentDelta(contentDelta);
    }

    return false;
}

async function getApiErrorMessage(response, fallbackMessage) {
    const text = await response.text();

    if (!text) {
        return fallbackMessage;
    }

    try {
        const data = JSON.parse(text);
        return data.error?.message || fallbackMessage;
    } catch {
        return text || fallbackMessage;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text ?? '';
    return div.innerHTML;
}

function formatSafeResponseHtml(escapedText) {
    // Input must already be HTML-escaped.
    // 1) Code fences ```...``` -> <pre><code>...</code></pre>
    // 2) Bold **...** -> <strong>...</strong>
    // 3) Newlines -> <br>
    let html = String(escapedText);

    html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code}</code></pre>`);
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\n/g, '<br>');
    return html;
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
                // Build DOM safely (no HTML injection from JSON).
                sidemenu.textContent = '';

                const nav = document.createElement('nav');
                nav.className = 'sidemenu';

                (Array.isArray(links) ? links : []).forEach(link => {
                    const a = document.createElement('a');
                    a.href = String(link?.href ?? '#');
                    a.textContent = String(link?.label ?? '');
                    nav.appendChild(a);
                });

                sidemenu.appendChild(nav);
            })
            .catch(() => {
                sidemenu.textContent = 'Menu failed to load';
            });
    });
}
