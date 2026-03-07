#!/bin/bash
# Generate ~/.config/opencode/opencode.json from environment variables at container startup
mkdir -p /root/.config/opencode
cat > /root/.config/opencode/opencode.json << EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "llama-cpp/${LLM_MODEL}",
  "provider": {
    "llama-cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp",
      "options": {
        "baseURL": "${LLM_BASE_URL}",
        "apiKey": "local"
      },
      "models": {
        "${LLM_MODEL}": {
          "name": "${LLM_MODEL}"
        }
      }
    }
  }
}
EOF

exec "$@"
