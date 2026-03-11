FROM docker.io/library/node:24-slim

# Install OpenCode CLI globally
RUN npm install -g opencode-ai

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
