#!/bin/bash
# Launch Multi-Panel Figure Builder as a native app
# Starts Python sidecar, then opens the Tauri .app

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../MultiPanelApp/.venv"
SIDECAR="$SCRIPT_DIR/python-sidecar/api_server.py"
TAURI_APP="$SCRIPT_DIR/src-tauri/target/release/bundle/macos/Multi-Panel Figure Builder.app"

# Activate venv
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

PORT=8765

# Kill any existing sidecar
pkill -f "api_server.py --port $PORT" 2>/dev/null
sleep 0.5

# Start Python sidecar in background
echo "Starting Python sidecar on port $PORT..."
python "$SIDECAR" --port "$PORT" &
SIDECAR_PID=$!

# Wait for sidecar to be ready
for i in {1..30}; do
    if curl -s "http://127.0.0.1:$PORT/api/config" > /dev/null 2>&1; then
        echo "Sidecar ready!"
        break
    fi
    sleep 0.3
done

# Open native app
if [ -d "$TAURI_APP" ]; then
    echo "Opening native app..."
    open "$TAURI_APP"
else
    echo "Native app not built. Opening in browser instead..."
    open "http://localhost:1420"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  Multi-Panel Figure Builder is running!"
echo "  API:     http://127.0.0.1:$PORT"
echo "  Press Ctrl+C to stop sidecar"
echo "═══════════════════════════════════════════"

# Wait for the Tauri app to close, then kill sidecar
trap "kill $SIDECAR_PID 2>/dev/null; exit" SIGINT SIGTERM
wait $SIDECAR_PID
