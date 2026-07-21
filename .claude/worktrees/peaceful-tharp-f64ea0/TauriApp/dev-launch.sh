#!/bin/bash
# Dev launch script with logging and restart capability
# Logs go to /tmp/multipanel-dev/

set -e

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 22.11.0
source "$HOME/.cargo/env"

LOGDIR="/tmp/multipanel-dev"
mkdir -p "$LOGDIR"

BACKEND_LOG="$LOGDIR/backend.log"
VITE_LOG="$LOGDIR/vite.log"
TAURI_LOG="$LOGDIR/tauri.log"

# Clean old logs
> "$BACKEND_LOG"
> "$VITE_LOG"
> "$TAURI_LOG"

echo "$(date): Starting dev environment..." | tee "$LOGDIR/launcher.log"

# Kill any existing processes
pkill -f "python.*api_server" 2>/dev/null || true
pkill -f "vite.*1420" 2>/dev/null || true
lsof -ti:1420 | xargs kill -9 2>/dev/null || true
lsof -ti:8765 | xargs kill -9 2>/dev/null || true
sleep 1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 1. Start Python backend on port 8765
echo "$(date): Starting Python backend on port 8765..." | tee -a "$LOGDIR/launcher.log"
cd "$SCRIPT_DIR/python-sidecar"
python3 api_server.py --port 8765 >> "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo "$(date): Backend PID=$BACKEND_PID" | tee -a "$LOGDIR/launcher.log"

# Wait for backend ready
for i in $(seq 1 15); do
    if grep -q "READY:8765" "$BACKEND_LOG" 2>/dev/null; then
        echo "$(date): Backend ready on port 8765" | tee -a "$LOGDIR/launcher.log"
        break
    fi
    sleep 1
done

# 2. Start Vite dev server
echo "$(date): Starting Vite on port 1420..." | tee -a "$LOGDIR/launcher.log"
cd "$SCRIPT_DIR"
npx vite --port 1420 >> "$VITE_LOG" 2>&1 &
VITE_PID=$!
echo "$(date): Vite PID=$VITE_PID" | tee -a "$LOGDIR/launcher.log"

# Wait for Vite ready
for i in $(seq 1 15); do
    if grep -q "ready in" "$VITE_LOG" 2>/dev/null; then
        echo "$(date): Vite ready on port 1420" | tee -a "$LOGDIR/launcher.log"
        break
    fi
    sleep 1
done

# 3. Launch Tauri native app
echo "$(date): Launching Tauri native app..." | tee -a "$LOGDIR/launcher.log"
cd "$SCRIPT_DIR"
npx tauri dev >> "$TAURI_LOG" 2>&1 &
TAURI_PID=$!
echo "$(date): Tauri PID=$TAURI_PID" | tee -a "$LOGDIR/launcher.log"

echo ""
echo "=== Dev environment started ==="
echo "Backend PID: $BACKEND_PID (log: $BACKEND_LOG)"
echo "Vite PID:    $VITE_PID (log: $VITE_LOG)"
echo "Tauri PID:   $TAURI_PID (log: $TAURI_LOG)"
echo "Launcher log: $LOGDIR/launcher.log"
echo ""
echo "To view logs: tail -f $LOGDIR/*.log"

# Wait for all processes
wait
