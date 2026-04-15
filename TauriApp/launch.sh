#!/bin/bash
# Launch Multi-Panel Figure Builder
# Starts the Python sidecar and opens the app in the browser

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../MultiPanelApp/.venv"
SIDECAR="$SCRIPT_DIR/python-sidecar/api_server.py"

# Activate venv
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Find a free port
PORT=8765

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

# Start Vite dev server
echo "Starting Vite dev server..."
cd "$SCRIPT_DIR"

# Find Node.js
if [ -f "$HOME/.nvm/versions/node/v22.11.0/bin/node" ]; then
    export PATH="$HOME/.nvm/versions/node/v22.11.0/bin:$PATH"
fi

npx vite --port 1420 &
VITE_PID=$!

# Wait for Vite
sleep 2

# Open in browser
echo "Opening app in browser..."
open "http://localhost:1420"

echo ""
echo "═══════════════════════════════════════════"
echo "  Multi-Panel Figure Builder is running!"
echo "  App:     http://localhost:1420"
echo "  API:     http://127.0.0.1:$PORT"
echo "  Press Ctrl+C to stop"
echo "═══════════════════════════════════════════"
echo ""

# Wait for Ctrl+C
trap "kill $SIDECAR_PID $VITE_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
