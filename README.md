# 🦙 llama.cpp on Podman

A production-ready containerized setup for running **llama.cpp inference server** with **Retrieval-Augmented Generation (RAG)** capabilities. Includes Apache HTTPD reverse proxy, ChromaDB vector store, a browser-based `aider` workspace, and an optional **MCP bridge** for safe, read-only PostgreSQL tools.

## ✨ Features

- 🦙 **LLM Inference** with llama.cpp and GGUF models
- 🧠 **RAG System** with semantic search and document Q&A
- 🔒 **Security** with HTTPS, authentication, and network isolation
- 🐳 **Containers** with a Podman kube stack
- 📁 **Multi-Format Support** for PDF, DOCX, TXT, and JSON documents
- 🧩 **MCP Bridge (optional)**: local-only, read-only PostgreSQL tools for the model
- 💻 **Browser Coding Workspace** with `aider --browser`

## 🚀 Quick Start

### 1. Setup

```bash
# Create local config files from examples
cp config/model-config.yaml.example config/model-config.yaml
cp config/postgresql-credentials.yaml.example config/postgresql-credentials.yaml

# Create required directories
mkdir -p \
    apache/certs \
    apache/logs \
    aider-workspace \
    models \
    rag-service/documents \
    rag-service/data \
    rag-service/models/embedding \
    rag-service/models/reranker \
    mcp-script/postgresql-data

# Set SELinux context (if SELinux is enabled on RHEL/Fedora/CentOS)
# chcon -Rt container_file_t ./rag-service/data/
# chcon -Rt container_file_t ./apache/certs/
# chcon -Rt container_file_t ./apache/conf/
# chcon -Rt container_file_t ./apache/html/
# chcon -Rt container_file_t ./apache/logs/
# chcon -Rt container_file_t ./models/

# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout apache/certs/server.key \
  -out apache/certs/server.crt \
  -subj "/C=US/ST=State/L=City/O=Org/OU=Unit/CN=localhost"
chmod 600 apache/certs/server.key
chmod 644 apache/certs/server.crt

# Create BASIC auth credentials expected by apache/conf/httpd.conf
python3 -c "
import crypt
username = 'user'
password = 'pass'
encrypted = crypt.crypt(password, crypt.mksalt(crypt.METHOD_SHA512))
print(f'{username}:{encrypted}')
" > apache/.htpasswd

# Build local images used by kube.yaml
podman build -t rag-service:latest -f rag-service/Dockerfile rag-service
podman build -t mcp-bridge:latest -f mcp-script/Dockerfile mcp-script
podman build -t aider-service:latest -f aider/Dockerfile aider
```

### 2. Start Services

```bash
# Production mode (with network isolation)
podman network create --internal isolated
./play-kube.sh

# Development mode (without isolation)
{
  cat config/model-config.yaml
  printf '\n---\n'
  cat config/postgresql-credentials.yaml
  printf '\n---\n'
  cat kube.yaml
} | podman play kube -
```

### 3. Access Web Interface

🌐 **Main UI**: https://localhost:8443/html/index.html
🧑‍💻 **Aider UI**: https://localhost:8443/aider/
🔐 **Login**: `user` / `pass`

## 🏗️ Architecture

```mermaid
graph TB
    A[Client Browser] -->|HTTPS:8443| B[Apache HTTPD]
    B -->|Proxy| C[llama.cpp Server]
    B -->|Proxy /rag/*| D[RAG Service]
    B -->|Proxy /aider/*| K[Aider Service]
    C -->|Model Inference| E[GGUF Models]
    D -->|Embeddings| F[llama.cpp embedding-service]
    D -->|Reranking| G[llama.cpp rerank-service]
    D -->|Vector Store| H[ChromaDB]
    D -->|LLM Calls| C
    J[MCP Bridge] -->|chat| C
    J -->|Read-only SQL tools| I[PostgreSQL]
    K -->|OpenAI-compatible calls| C
    
    subgraph "Container Network"
        B
        C
        D
        E
        F
        G
        H
        I
        J
        K
    end
```

## 📁 Project Structure

```
llama-cpp-podman/
├── 🐳 Container Configs
│   ├── kube.yaml             # Podman kube manifest
│   └── config/
│       ├── model-config.yaml.example
│       ├── model-config.yaml  # model settings (excluded from git)
│       ├── postgresql-credentials.yaml.example
│       └── postgresql-credentials.yaml  # postgres settings (excluded from git)
├── 🌐 Apache Setup
│   ├── conf/
│   │   └── httpd.conf        # Reverse proxy + security
│   ├── .htpasswd             # Authentication (excluded from git)
│   ├── certs/                # SSL certificates (excluded from git)
│   ├── logs/                 # Access logs
│   └── html/                 # Web applications
│       ├── index.html        # Main web interface
│       ├── rag.html          # RAG document Q&A
│       ├── rag-chat.html     # RAG continuous chat
│       ├── translate.html    # JA ⇔ EN translation
│       ├── proofread.html    # Text proofreading
│       ├── reply.html        # Email assistant
│       ├── mcp.html          # MCP bridge chat interface
│       ├── sidemenu.html     # Navigation menu
│       ├── css/              # Stylesheets
│       └── js/               # Client-side functionality
├── 🤖 Models
│   └── *.gguf                # Main llama.cpp + mmproj GGUF files
├── 🧠 RAG Service
│   ├── src/
│   │   ├── main.py           # FastAPI application
│   │   ├── rag_chain.py      # RAG implementation
│   │   ├── vector_store.py   # ChromaDB integration
│   │   ├── custom_embeddings.py
│   │   ├── reranker.py
│   │   ├── document_loader.py
│   │   └── http_session.py   # Shared HTTP session (connection pooling + retries)
│   ├── documents/            # Input documents (excluded from git)
│   ├── data/                 # Vector database (excluded from git)
│   ├── models/               # Embedding and rerank models (excluded from git)
│   ├── requirements.txt
│   └── Dockerfile
├── 🧩 MCP Bridge
│   ├── bridge.py             # CLI bridge (llama.cpp ↔ MCP)
│   ├── web_bridge.py         # FastAPI bridge (HTTP)
│   ├── pg_server.py          # MCP server exposing read-only PG tools
│   ├── postgresql/init.sql   # Example schema/data loaded on first init
│   └── postgresql-data/      # Postgres persistent data (hostPath)
├── 💻 aider
│   ├── Dockerfile            # Aider browser image
│   └── entrypoint.sh
├── aider-workspace/                  # Workspace mounted into aider-service at /workspace
└── 📚 Documentation
    ├── README.md             # This file
    └── LICENSE
```

## 🛠️ Operations

### Starting and Stopping Services

```bash
# Stop services
./stop-kube.sh

# Restart specific service
podman restart rag-service-deployment-pod-rag-service
podman restart llama-cpp-server-deployment-pod-llama-cpp-server
```

### Monitoring and Logs

```bash
# Status overview
./scripts/monitor.sh

# Follow logs for one service
./scripts/monitor.sh logs apache

# Show recent logs without follow
./scripts/monitor.sh tail postgres 200

# Show service aliases
./scripts/monitor.sh list

# Health checks
./scripts/monitor.sh health
```

`scripts/monitor.sh` wraps the common `podman ps`, `podman logs -f`, and HTTPS health checks.
Override defaults with `MONITOR_BASE_URL`, `BASIC_AUTH_USER`, and `BASIC_AUTH_PASS` when needed.

### kube.yaml integration

[kube.yaml](kube.yaml) defines the full Podman kube stack, including the optional `postgresql` and `mcp-bridge` components, Apache, the RAG service, and `aider-service`.
Shared settings are in [config/model-config.yaml](config/model-config.yaml), local PostgreSQL credentials are read from `config/postgresql-credentials.yaml`, and `play-kube.sh` / `stop-kube.sh` stream those files together with `kube.yaml` into `podman play kube`.

The `aider-service` container starts `aider --browser` automatically and is exposed through Apache at `https://localhost:8443/aider/`. It mounts [aider-workspace/](aider-workspace/) into `/workspace`, and its entrypoint will initialize a git repo there automatically when browser mode needs one. By default it reuses `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `LLM_MODEL` from the shared config map so it can talk to the in-cluster llama.cpp OpenAI-compatible endpoint.
If you want `aider` to call a different OpenAI-compatible provider, override `AIDER_OPENAI_API_KEY`, `AIDER_OPENAI_API_BASE`, or `AIDER_MODEL`. If that provider is outside the isolated network, start the stack without `--network isolated` or relax the `aider-service` egress policy.

- PostgreSQL initialization:
    - [mcp-script/postgresql/init.sql](mcp-script/postgresql/init.sql) is mounted into `/docker-entrypoint-initdb.d/00-init.sql`.
    - It runs **only on first init** (when the data directory is empty).
    - To re-apply, delete the hostPath directory [mcp-script/postgresql-data/](mcp-script/postgresql-data/) and restart.

## ⚙️ Configuration

### Model Configuration

Update `config/model-config.yaml` with your model settings:

```yaml
kind: ConfigMap
metadata:
    name: model-config
data:
    LLM_MODEL_NAME: Qwen3-0.6B-Q8_0
```

The RAG service uses these key environment variables (see `config/model-config.yaml`):
- `LLAMA_CPP_BASE_URL` (default: `http://llama-cpp-server:11434`)
- `LLM_MODEL_NAME` (GGUF basename used by `kube.yaml`)
- `LLM_MODEL` (model id from llama.cpp `/v1/models`)
- `EMBEDDING_API_BASE` / `EMBEDDING_MODEL_NAME`
- `RERANK_API_BASE` / `RERANK_MODEL_NAME`
- `RETRIEVAL_CANDIDATES` / `RERANK_TOP_K`

The MCP bridge uses these key environment variables (see `config/model-config.yaml`; for host runs, see [mcp-script/.env.template](mcp-script/.env.template)):

- `LLM_BASE_URL` (example in-cluster: `http://llama-cpp-server:11434/v1`)
- `OPENAI_API_BASE` (alias for OpenAI-compatible clients like `aider`)
- `LLM_MODEL` (model id from `GET /v1/models`)
- `OPENAI_API_KEY` (use `local` for llama.cpp)
- `PG_HOST`, `PG_PORT`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD` (or use `PG_DSN` when running on the host)

The browser-based `aider-service` also honors these overrides:

- `AIDER_OPENAI_API_KEY`
- `AIDER_OPENAI_API_BASE`
- `AIDER_MODEL`
- `AIDER_EXTRA_ARGS`

## 🧠 RAG System Deep Dive

### Pipeline Architecture

1. **Document Ingestion** → Text splitting → Embedding generation → Vector storage
2. **Query Processing** → Vector search → Reranking → Context generation → LLM response

### Models Used

- **Embedding**: GGUF served by llama.cpp embedding-service (`/v1/embeddings`)
- **Reranker**: GGUF served by llama.cpp rerank-service (`/v1/rerank`)
- **LLM**: Your GGUF model via llama.cpp

### Dependencies

The RAG service uses LangChain 1.1.x with modular packages (see [rag-service/requirements.txt](rag-service/requirements.txt)):
- `langchain-classic`, `langchain-core`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`
- `chromadb`, `fastapi`, `uvicorn`, `requests`, `openai`, `python-multipart`
- `pypdf`, `docx2txt` for document parsing

### API Reference

<details>
<summary>Click to expand API documentation</summary>

#### RAG Endpoints

```bash
# System Status
GET /rag/status
Response: {"status": "ready", "vector_store": {"document_count": 42, ...}, "llm_available": true, "chain_ready": true}

# Query with RAG
POST /rag/query
Body: {"question": "What is vector search?", "include_sources": true}
Response: {"answer": "...", "sources": [...]}

# Document Search
POST /rag/search  
Body: {"question": "machine learning", "k": 5}
Response: [{"content": "...", "metadata": {...}, "similarity_score": 87.0, ...}]

# Upload Documents
POST /rag/upload
Form: files=@document.pdf
Response: {"message": "Successfully uploaded and processed 1 files", "documents_added": 12, "file_names": ["document.pdf"]}

# Bulk Ingest
POST /rag/ingest
Form: directory_path=/app/documents
Response: {"message": "Successfully ingested documents", "documents_added": 12}

# Clear Database
DELETE /rag/documents
Response: {"message": "All documents cleared successfully"}

# Delete by content
POST /rag/documents/delete-by-content
Body: {"content_query": "partial text"}
Response: {"deleted_count": 1, ...}
```

#### LLM Endpoints

```bash
# Text Completion
POST /completion
Body: {"prompt": "Explain AI", "max_tokens": 256}
Response: {"content": "..."}

# Chat Completion  
POST /v1/chat/completions
Body: {"messages": [{"role": "user", "content": "Hello"}]}
Response: {"choices": [...]}
```

</details>

### Document Formats

| Format | Support | Notes |
|--------|---------|-------|
| **PDF** | ✅ Full | Text extraction, metadata preserved |
| **DOCX** | ✅ Full | Word documents, formatting stripped |
| **TXT** | ✅ Full | Plain text files |
| **JSON** | ✅ Partial | Format: `{"chunks": ["text1", "text2"]}` |

## 🔒 Security Configuration

### Development Environment

**Default Settings:**
- Self-signed SSL certificates
- Basic authentication: `user` / `pass`
- CORS: Disabled by default (same-origin; Apache proxies UI and API together)
- Network: Internet access enabled

### Production Hardening

**1. SSL Certificates:**
```bash
# Use proper certificates
cp your-cert.crt apache/certs/server.crt
cp your-key.key apache/certs/server.key
chmod 600 apache/certs/server.key
```

**2. Authentication:**
```bash
# Generate password hash (change username/password as needed)
python3 -c "
import crypt
username = 'user'
password = 'pass'
encrypted = crypt.crypt(password, crypt.mksalt(crypt.METHOD_SHA512))
print(f'{username}:{encrypted}')
" > apache/.htpasswd  # Use >> to append additional users
```

**3. CORS Configuration:**

Set `CORS_ALLOW_ORIGINS` in `config/model-config.yaml` (comma-separated):
```yaml
CORS_ALLOW_ORIGINS: "https://yourdomain.com"
```
Wildcard with credentials is never allowed. Omit the variable to keep CORS disabled (default).

**4. Network Isolation:**
Use `./play-kube.sh` (see Quick Start).

**5. Firewall Rules:**
```bash
sudo ufw allow 8443/tcp   # HTTPS only
sudo ufw deny 8080/tcp    # Block HTTP
sudo ufw deny 11434/tcp   # Block direct LLM access
```

**Security Checklist:**
- [ ] SSL certificates with proper permissions (600 for .key)
- [ ] Strong authentication credentials
- [ ] CORS configured for specific origins
- [ ] Network isolation enabled
- [ ] Firewall rules configured
- [ ] Regular security updates
- [ ] Log monitoring enabled

## 🌟 Advanced Usage

### Custom Model Integration

```bash
# Add new model
cp your-model.gguf models/

# Restart services
./stop-kube.sh
./play-kube.sh
```

### Bulk Document Processing

```bash
# Upload directory of documents
for file in /path/to/docs/*; do
    curl -k -u user:pass -X POST https://localhost:8443/rag/upload \
         -F "files=@$file"
done

# Or use the ingest endpoint for documents/ directory
curl -k -u user:pass -X POST https://localhost:8443/rag/ingest \
    -F "directory_path=/app/documents"
```

### API Integration Examples

<details>
<summary>Python Client Example</summary>

```python
import requests
from requests.auth import HTTPBasicAuth

class LlamaRAGClient:
    def __init__(self, base_url="https://localhost:8443", username="user", password="pass"):
        self.base_url = base_url
        self.auth = HTTPBasicAuth(username, password)
        self.session = requests.Session()
        self.session.verify = False  # For self-signed certs

    def query(self, question, include_sources=True):
        response = self.session.post(
            f"{self.base_url}/rag/query",
            json={"question": question, "include_sources": include_sources},
            auth=self.auth
        )
        return response.json()

    def upload_document(self, file_path):
        with open(file_path, 'rb') as f:
            response = self.session.post(
                f"{self.base_url}/rag/upload",
                files={"files": f},
                auth=self.auth
            )
        return response.json()

# Usage
client = LlamaRAGClient()
result = client.query("What is machine learning?")
print(result["answer"])
```

</details>

## 🚨 Troubleshooting

### Common Issues

<details>
<summary>🔧 Service Won't Start</summary>

```bash
# Check pod status
podman pod ps

# Check container logs
./scripts/monitor.sh tail rag

# Rebuild RAG service
cd rag-service
podman build -t rag-service:latest .
./stop-kube.sh
./play-kube.sh
```

</details>

<details>
<summary>🧩 MCP / PostgreSQL Issues</summary>

```bash
# Check Postgres boot + init.sql ran (only on first init)
./scripts/monitor.sh tail postgres 200

# If you need to re-apply init.sql, delete persistent data and restart
rm -rf mcp-script/postgresql-data
mkdir -p mcp-script/postgresql-data
./stop-kube.sh
./play-kube.sh

```

</details>

<details>
<summary>🔐 Authentication Issues</summary>

```bash
# Verify password file exists and has content
cat apache/.htpasswd

# Regenerate credentials (see Security Configuration section above)
# Test authentication
curl -k -u user:pass https://localhost:8443/health
```

</details>

<details>
<summary>🤖 Model Loading Problems</summary>

```bash
# Check model file exists
ls -la models/

# Verify model configuration in kube.yaml
grep -E "LLM_MODEL|LLM_MODEL‗NAME|EMBEDDING_MODEL_NAME|RERANK_MODEL_NAME" kube.yaml

# Check llama.cpp logs for errors
./scripts/monitor.sh tail llm 200 | grep -i error

</details>

<details>
<summary>🧠 RAG Issues</summary>

```bash
# Check embedding models
ls -la rag-service/models/embedding/
ls -la rag-service/models/reranker/

# Test RAG service directly
curl -k -u user:pass https://localhost:8443/rag/status

# Clear and rebuild vector store
rm -rf rag-service/data/chroma_db/
# Re-upload documents
```

</details>

### Performance Optimization

| Component | Memory | CPU | Storage | Notes |
|-----------|--------|-----|---------|-------|
| **llama.cpp** | 4-16GB | 4+ cores | - | Depends on model size |
| **RAG Service** | 2-4GB | 2+ cores | SSD preferred | Embedding models |
| **ChromaDB** | 1-2GB | 1+ core | SSD required | Vector operations |
| **Apache** | 512MB | 1 core | - | Reverse proxy |

**Optimization Tips:**
- Use SSD for vector database performance
- Increase `n_predict` for longer responses
- Tune `temperature` for creativity vs consistency
- Monitor memory usage during operation

## 📄 License

This project is licensed under the MIT License.
