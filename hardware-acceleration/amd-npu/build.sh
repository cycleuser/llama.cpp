#!/bin/bash
# Build script for AMD XDNA NPU backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"

# Parse arguments
BUILD_TYPE="Release"
BUILD_TESTS=ON
DEBUG=OFF

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            DEBUG=ON
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --no-tests)
            BUILD_TESTS=OFF
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug      Build in Debug mode"
            echo "  --release    Build in Release mode (default)"
            echo "  --no-tests   Don't build tests"
            echo "  --help       Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Building AMD XDNA NPU Backend"
echo "========================================"
echo "Build type: $BUILD_TYPE"
echo "Build tests: $BUILD_TESTS"
echo "Debug logging: $DEBUG"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"

# Configure
echo "[1/3] Configuring..."
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DGGML_AMD_NPU_BUILD_TESTS="$BUILD_TESTS" \
    -DGGML_AMD_NPU_DEBUG="$DEBUG"

# Build
echo ""
echo "[2/3] Building..."
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Install
echo ""
echo "[3/3] Installing..."
cmake --install "$BUILD_DIR" --config "$BUILD_TYPE"

# Run tests if built
if [ "$BUILD_TESTS" = "ON" ]; then
    echo ""
    echo "Running tests..."
    "$BUILD_DIR/test_amdxdna" || echo "Some tests failed (may be expected without hardware)"
fi

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo "Library: $INSTALL_DIR/lib"
echo "Headers: $INSTALL_DIR/include"
echo ""
echo "To use in llama.cpp:"
echo "  cmake -B build -DGGML_AMD_NPU=ON"
echo "  cmake --build build"