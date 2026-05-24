#!/bin/bash
# Build the ScreenCaptureKit window-recorder helper (macOS only) and place it in
# src-tauri/binaries with the target-triple suffix Tauri's externalBin expects.
# Usage: ./scripts/build-recorder.sh [target-triple]
# NOTE: the committed tauri.conf.json does NOT list binaries/mpfb-recorder in
# externalBin (so Windows/local non-mac builds don't need it). To bundle it into
# a local `tauri build`, add "binaries/mpfb-recorder" to externalBin after
# running this (the CI does that injection automatically).
set -e

if [ "$(uname)" != "Darwin" ]; then
  echo "mpfb-recorder is macOS-only (ScreenCaptureKit). Skipping on $(uname)."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC="$PROJECT_DIR/native-helpers/mpfb-recorder/recorder.swift"
BIN_DIR="$PROJECT_DIR/src-tauri/binaries"

if [ -n "$1" ]; then TRIPLE="$1"; else TRIPLE=$(rustc -vV | grep 'host:' | awk '{print $2}'); fi
mkdir -p "$BIN_DIR"
OUT="$BIN_DIR/mpfb-recorder-${TRIPLE}"

echo "Building mpfb-recorder for: $TRIPLE"
swiftc -O -swift-version 5 -target arm64-apple-macos14.0 \
  -framework ScreenCaptureKit -framework AVFoundation \
  -framework CoreMedia -framework CoreGraphics \
  -o "$OUT" "$SRC"
chmod +x "$OUT"
echo "✅ Recorder helper built: $OUT"
file "$OUT"
