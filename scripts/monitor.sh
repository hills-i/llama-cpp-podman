#!/bin/bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
BASE_URL="${MONITOR_BASE_URL:-https://localhost:8443}"
BASIC_AUTH_USER="${BASIC_AUTH_USER:-user}"
BASIC_AUTH_PASS="${BASIC_AUTH_PASS:-pass}"

declare -A SERVICES=(
    [llm]="llama-cpp-server-deployment-pod-llama-cpp-server"
    [rag]="rag-service-deployment-pod-rag-service"
    [embedding]="embedding-service-deployment-pod-embedding-service"
    [rerank]="rerank-service-deployment-pod-rerank-service"
    [apache]="apache-deployment-pod-apache"
    [postgres]="postgresql-deployment-pod-postgresql"
    [mcp]="mcp-bridge-deployment-pod-mcp-bridge"
    [aider]="aider-service-deployment-pod-aider-service"
)

print_usage() {
    cat <<EOF
Usage:
  ./$SCRIPT_NAME [status]
  ./$SCRIPT_NAME logs <service>
  ./$SCRIPT_NAME tail <service> [lines]
  ./$SCRIPT_NAME health
  ./$SCRIPT_NAME list

Services:
  llm, rag, embedding, rerank, apache, postgres, mcp, aider

Environment variables:
  MONITOR_BASE_URL   Default: $BASE_URL
  BASIC_AUTH_USER    Default: $BASIC_AUTH_USER
  BASIC_AUTH_PASS    Default: <hidden>
EOF
}

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: '$command_name' is not installed." >&2
        exit 1
    fi
}

resolve_service() {
    local service_name="${1:-}"

    if [[ -z "$service_name" ]]; then
        echo "Error: service name is required." >&2
        print_usage >&2
        exit 1
    fi

    if [[ -n "${SERVICES[$service_name]:-}" ]]; then
        printf '%s\n' "${SERVICES[$service_name]}"
        return 0
    fi

    printf '%s\n' "$service_name"
}

show_status() {
    require_command podman
    podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

show_logs() {
    require_command podman
    local container_name
    container_name="$(resolve_service "${1:-}")"
    podman logs -f "$container_name"
}

show_recent_logs() {
    require_command podman
    local container_name
    local lines="${2:-200}"

    container_name="$(resolve_service "${1:-}")"
    podman logs --tail "$lines" "$container_name"
}

show_health_check() {
    local label="$1"
    local url="$2"
    local auth="$3"
    local status

    status="$(curl --fail --silent --show-error -k -u "$auth" \
        -o /dev/null -w '%{http_code}' "$url")"

    printf '[%s] %s -> HTTP %s\n' "$label" "$url" "$status"
}

show_health() {
    require_command curl

    local auth="${BASIC_AUTH_USER}:${BASIC_AUTH_PASS}"

    show_health_check "apache" "$BASE_URL/health" "$auth"
    show_health_check "rag" "$BASE_URL/rag/status" "$auth"
    show_health_check "aider" "$BASE_URL/aider/" "$auth"
}

list_services() {
    local service_name
    for service_name in llm rag embedding rerank apache postgres mcp aider; do
        printf '%-10s %s\n' "$service_name" "${SERVICES[$service_name]}"
    done
}

main() {
    local command="${1:-status}"

    case "$command" in
        status)
            show_status
            ;;
        logs)
            show_logs "${2:-}"
            ;;
        tail)
            show_recent_logs "${2:-}" "${3:-200}"
            ;;
        health)
            show_health
            ;;
        list)
            list_services
            ;;
        help|-h|--help)
            print_usage
            ;;
        *)
            echo "Error: unknown command '$command'." >&2
            print_usage >&2
            exit 1
            ;;
    esac
}

main "$@"
