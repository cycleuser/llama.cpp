#!/bin/bash
# =============================================================================
# Ollama Model Manager for llama.cpp
# =============================================================================
#
# Commands:
#   ./ollama.sh list              - List all installed models
#   ./ollama.sh show <model>      - Show model details
#   ./ollama.sh run <model>       - Run model with llama-cli
#   ./ollama.sh bench <model>     - Benchmark model
#
# Examples:
#   ./ollama.sh list
#   ./ollama.sh show library/gemma3:1b
#   ./ollama.sh run library/gemma3:1b -p "Hello"
#   ./ollama.sh bench library/gemma3:1b
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

OLLAMA_DIR="${OLLAMA_MODELS:-$HOME/.ollama/models}"
MANIFESTS="$OLLAMA_DIR/manifests/registry.ollama.ai"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

get_gguf_path() {
    local manifest="$1"
    local digest=$(jq -r '.layers[] | select(.mediaType == "application/vnd.ollama.image.model") | .digest' "$manifest" 2>/dev/null | head -1)
    digest="${digest#sha256:}"
    if [ -n "$digest" ]; then
        echo "$OLLAMA_DIR/blobs/sha256-$digest"
    fi
}

get_llama_cli() {
    local builds=(
        "$ROOT_DIR/build-metal/bin/llama-cli"
        "$ROOT_DIR/build-cuda/bin/llama-cli"
        "$ROOT_DIR/build-hip/bin/llama-cli"
        "$ROOT_DIR/build-vulkan/bin/llama-cli"
        "$ROOT_DIR/build/bin/llama-cli"
    )
    for build in "${builds[@]}"; do
        [ -x "$build" ] && echo "$build" && return 0
    done
    return 1
}

list_models() {
    echo -e "${CYAN}${BOLD}Installed Ollama Models${NC}"
    echo "======================="
    echo ""
    
    local count=0
    find "$MANIFESTS" -type f 2>/dev/null | sort | while read manifest; do
        path="${manifest#$MANIFESTS/}"
        namespace=$(echo "$path" | cut -d'/' -f1)
        model=$(echo "$path" | cut -d'/' -f2)
        tag=$(echo "$path" | cut -d'/' -f3)
        
        digest=$(jq -r '.layers[] | select(.mediaType == "application/vnd.ollama.image.model") | .digest' "$manifest" 2>/dev/null | head -1)
        digest="${digest#sha256:}"
        
        if [ -n "$digest" ] && [ -f "$OLLAMA_DIR/blobs/sha256-$digest" ]; then
            size=$(stat -f%z "$OLLAMA_DIR/blobs/sha256-$digest" 2>/dev/null || stat -c%s "$OLLAMA_DIR/blobs/sha256-$digest" 2>/dev/null)
            size_gb=$(echo "scale=1; $size / 1073741824" | bc 2>/dev/null || echo "$((size / 1073741824))")
            printf "${GREEN}%-35s${NC} %5s GB\n" "$namespace/$model:$tag" "$size_gb"
        fi
    done
}

show_model() {
    local model_spec="$1"
    
    if [ -z "$model_spec" ]; then
        echo -e "${RED}Usage: $0 show <namespace/model:tag>${NC}"
        return 1
    fi
    
    local namespace="library"
    local model=""
    local tag="latest"
    
    if [[ "$model_spec" == *"/"* ]]; then
        namespace="${model_spec%%/*}"
        model_spec="${model_spec#*/}"
    fi
    
    if [[ "$model_spec" == *":"* ]]; then
        model="${model_spec%:*}"
        tag="${model_spec#*:}"
    else
        model="$model_spec"
    fi
    
    local manifest="$MANIFESTS/$namespace/$model/$tag"
    
    if [ ! -f "$manifest" ]; then
        echo -e "${RED}Model not found: $namespace/$model:$tag${NC}"
        return 1
    fi
    
    echo -e "${CYAN}${BOLD}Model: $namespace/$model:$tag${NC}"
    echo "================================"
    echo ""
    echo -e "${BOLD}Manifest:${NC} $manifest"
    echo ""
    echo -e "${BOLD}Layers:${NC}"
    jq -r '.layers[] | "  \(.mediaType | split("/")[-1]): \(.digest) (\(.size | tostring)s)"' "$manifest" 2>/dev/null || grep -E '"mediaType"|"digest"|"size"' "$manifest"
    echo ""
    
    local gguf=$(get_gguf_path "$manifest")
    echo -e "${BOLD}GGUF Path:${NC} $gguf"
    
    if [ -n "$gguf" ] && [ -f "$gguf" ]; then
        local size=$(stat -f%z "$gguf" 2>/dev/null || stat -c%s "$gguf" 2>/dev/null)
        local size_human=$(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "$size bytes")
        echo -e "${BOLD}File Size:${NC} $size_human"
    fi
}

run_model() {
    local model_spec="$1"
    shift
    
    local namespace="library"
    local model=""
    local tag="latest"
    
    if [[ "$model_spec" == *"/"* ]]; then
        namespace="${model_spec%%/*}"
        model_spec="${model_spec#*/}"
    fi
    
    if [[ "$model_spec" == *":"* ]]; then
        model="${model_spec%:*}"
        tag="${model_spec#*:}"
    else
        model="$model_spec"
    fi
    
    local manifest="$MANIFESTS/$namespace/$model/$tag"
    
    if [ ! -f "$manifest" ]; then
        echo -e "${RED}Model not found: $namespace/$model:$tag${NC}"
        return 1
    fi
    
    local gguf=$(get_gguf_path "$manifest")
    
    if [ -z "$gguf" ] || [ ! -f "$gguf" ]; then
        echo -e "${RED}GGUF file not found${NC}"
        return 1
    fi
    
    local llama_cli=$(get_llama_cli)
    
    if [ -z "$llama_cli" ]; then
        echo -e "${RED}No llama.cpp build found. Run ./scripts/build.sh first.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Running model: $namespace/$model:$tag${NC}"
    echo -e "${YELLOW}Using: $llama_cli${NC}"
    echo ""
    
    "$llama_cli" -m "$gguf" "$@"
}

bench_model() {
    local model_spec="$1"
    shift
    
    local namespace="library"
    local model=""
    local tag="latest"
    
    if [[ "$model_spec" == *"/"* ]]; then
        namespace="${model_spec%%/*}"
        model_spec="${model_spec#*/}"
    fi
    
    if [[ "$model_spec" == *":"* ]]; then
        model="${model_spec%:*}"
        tag="${model_spec#*:}"
    else
        model="$model_spec"
    fi
    
    local manifest="$MANIFESTS/$namespace/$model/$tag"
    
    if [ ! -f "$manifest" ]; then
        echo -e "${RED}Model not found: $namespace/$model:$tag${NC}"
        return 1
    fi
    
    local gguf=$(get_gguf_path "$manifest")
    
    if [ -z "$gguf" ] || [ ! -f "$gguf" ]; then
        echo -e "${RED}GGUF file not found${NC}"
        return 1
    fi
    
    local llama_bench="$ROOT_DIR/build-metal/bin/llama-bench"
    [ ! -x "$llama_bench" ] && llama_bench="$ROOT_DIR/build/bin/llama-bench"
    [ ! -x "$llama_bench" ] && llama_bench="$ROOT_DIR/build-cuda/bin/llama-bench"
    
    if [ -z "$llama_bench" ] || [ ! -x "$llama_bench" ]; then
        echo -e "${RED}llama-bench not found${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Benchmarking: $namespace/$model:$tag${NC}"
    echo -e "${YELLOW}GGUF: $gguf${NC}"
    echo ""
    
    "$llama_bench" -m "$gguf" "$@"
}

main() {
    local command="${1:-list}"
    shift || true
    
    case "$command" in
        list)  list_models ;;
        show)  show_model "$@" ;;
        run)   run_model "$@" ;;
        bench) bench_model "$@" ;;
        *)
            echo "Usage: $0 {list|show|run|bench} [args...]"
            echo ""
            echo "Commands:"
            echo "  list              List all installed models"
            echo "  show <model>      Show model details"
            echo "  run <model>       Run model with llama-cli"
            echo "  bench <model>     Benchmark model"
            ;;
    esac
}

main "$@"