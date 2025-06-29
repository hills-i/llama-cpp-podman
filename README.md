# llama-cpp Podman Setup

This provides a setup for running [llama.cpp](https://github.com/ggml-org/llama.cpp) and Apache HTTPD using Podman. It supports running the containers with or without internet access.

## Features

- **Podman**: Easily run and manage containers using `podman-compose` or `podman play kube`.
- **No Internet Access Option**: The internal network can be configured to restrict internet access for containers in accordance with the organization's security policies.
- **Apache Reverse Proxy**: Apache serves as a reverse proxy (with HTTPS) to the llama.cpp server.
- **Persistent Volumes**: Models, logs, configs, and certificates are mounted from the host.

## Quick Start

### 1. Prerequisites

- Podman installed.
  (Podman-compose installed.)
- Model file in `./models` directory (e.g., `*.gguf`).
- Apache config, certs, and HTML files in `./apache/conf`, `./apache/certs`, `./apache/html`.

## Configuration: Setting the Model File Name

The environment variable `MODEL_FILE` specifies the name of your model file (e.g., `/models/model.gguf`).  
You must set this variable according to your file in `podman-compose.yml` or `kube.yaml`.

- In `podman-compose.yml`, set:
  ```yaml
  environment:
    - MODEL_FILE=/models/model.gguf
  ```
- In `kube.yaml`, set:
  ```yaml
  env:
    - name: MODEL_FILE
      value: /models/model.gguf
  ```

Replace `model.gguf` with the actual path to your model file on your system.

### 2. Start the Services

#### Using Podman Compose

```sh
podman-compose -f podman-compose.yml up
```

- Apache will be available on ports `8080` (HTTP) and `8443` (HTTPS).
- llama.cpp server will be available on port `11434` (internal).

#### Using Podman play kube

You can also use Kubernetes YAML with Podman:

```sh
# Create an internal network (no internet access)
podman network create --internal isolated

# Start the services using the isolated network
podman play kube --network isolated kube.yaml
```

- This will create pods and services as defined in `kube.yaml` on the isolated network.
- To stop and remove all resources:

```sh
podman play kube --down kube.yaml
```

### 3. Network Configuration

#### Default: No Internet Access for Internal Network

By default, the `internal` network is set as `internal: true` in `podman-compose.yml`:

```yaml
networks:
  internal:
    driver: bridge
    internal: true
```

This means containers on the `internal` network **cannot access the internet**. Only communication between containers on this network is allowed.

- For `podman play kube`, you can restrict internet access by using NetworkPolicy in `kube.yaml`, or by running with an internal Podman network as shown above.

#### Allow Internet Access

To allow internet access for containers, set `internal: false` or remove the `internal` line:

```yaml
networks:
  internal:
    driver: bridge
    # internal: true   # Comment out or remove this line
```

Then restart the containers:

```sh
podman-compose down
podman-compose up
```

### 4. Stopping the Services

```sh
podman-compose down
```

## Directory Structure

```
llama-cpp/
├── apache/
│   ├── conf/      # Apache config (httpd.conf)
│   ├── certs/     # SSL certificates
│   ├── logs/      # Apache logs
│   └── html/      # Static HTML files
├── models/        # Llama model files
├── podman-compose.yml
└── README.md
```

## Sample Applications

The `apache/html` directory contains sample web applications.

## How to Generate SSL Certificates

You can generate a self-signed SSL certificate for development/testing:

```sh
mkdir -p apache/certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout apache/certs/server.key \
  -out apache/certs/server.crt \
  -subj "/C=US/ST=State/L=City/O=Org/OU=Unit/CN=localhost"
```

## How to Create .htpasswd for BASIC Auth

To enable BASIC authentication, create a `.htpasswd` file:

```sh
# Install apache2-utils if not present (Debian/Ubuntu)
sudo apt-get install apache2-utils

# Or install httpd-tools (Fedora/CentOS)
sudo dnf install httpd-tools

# Create .htpasswd with a user (replace USERNAME as needed)
htpasswd -c apache/conf/.htpasswd USERNAME
```

- You will be prompted to enter a password.
- To add more users, omit the `-c` flag.

Uncomment the relevant `<Location />` block in `apache/conf/httpd.conf` to enable BASIC auth.

## Security Notes

- For additional security, you can run containers with no internet access using Podman's `--internal` network option.  
  This is not because we distrust llama.cpp, but as a best practice for minimizing risk.
- The Apache server is configured to use HTTPS and reverse proxy to the llama.cpp server.
- Adjust Apache and network settings as needed for your environment.

## References

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Podman Compose](https://github.com/containers/podman-compose)

# License

MIT License
