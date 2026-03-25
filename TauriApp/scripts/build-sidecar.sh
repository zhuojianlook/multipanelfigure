#!/bin/bash
# Build the Python sidecar as a standalone binary using PyInstaller
# Usage: ./scripts/build-sidecar.sh [target-triple]
# If no target-triple, uses the current platform

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SIDECAR_DIR="$PROJECT_DIR/python-sidecar"
BIN_DIR="$PROJECT_DIR/src-tauri/binaries"

# Determine target triple
if [ -n "$1" ]; then
    TRIPLE="$1"
else
    TRIPLE=$(rustc -vV | grep 'host:' | awk '{print $2}')
fi

echo "Building sidecar for: $TRIPLE"
echo "Source: $SIDECAR_DIR/api_server.py"

cd "$SIDECAR_DIR"

# Build with PyInstaller
pyinstaller --onefile \
    --name "api-server" \
    --hidden-import=uvicorn.logging \
    --hidden-import=uvicorn.protocols.http \
    --hidden-import=uvicorn.protocols.http.auto \
    --hidden-import=uvicorn.protocols.websockets \
    --hidden-import=uvicorn.protocols.websockets.auto \
    --hidden-import=uvicorn.lifespan \
    --hidden-import=uvicorn.lifespan.on \
    --hidden-import=uvicorn.lifespan.off \
    --collect-all PIL \
    --collect-all matplotlib \
    --collect-all cv2 \
    --noconfirm \
    api_server.py

# Copy to Tauri binaries directory
mkdir -p "$BIN_DIR"
cp "dist/api-server" "$BIN_DIR/api-server-${TRIPLE}"
chmod +x "$BIN_DIR/api-server-${TRIPLE}"

echo ""
echo "✅ Sidecar built: $BIN_DIR/api-server-${TRIPLE}"
ls -lh "$BIN_DIR/api-server-${TRIPLE}"
