FROM docker.io/library/python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy scripts
COPY bridge.py /app/bridge.py
COPY pg_server.py /app/pg_server.py
COPY web_bridge.py /app/web_bridge.py

# Default: interactive bridge (exec into pod for TTY)
EXPOSE 8090

CMD ["uvicorn", "web_bridge:app", "--host", "0.0.0.0", "--port", "8090"]
