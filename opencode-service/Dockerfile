FROM docker.io/node:24-slim

# Install system tools for development
RUN apt-get update && apt-get install -y --no-install-recommends \
    git vim nano curl python3 python3-pip bash \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCode CLI globally
RUN npm install -g opencode-ai

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
