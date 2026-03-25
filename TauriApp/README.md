# Multi-Panel Figure Builder

A professional desktop application for creating multi-panel scientific figures with full control over layout, annotations, scale bars, and image adjustments. Built with Tauri (Rust + React) and a Python image processing backend.

## Features

### Panel Layout
- Configurable grid layout (rows x columns) with drag-and-drop image assignment
- Panel reordering with settings preservation
- Parking drawer for temporary panel storage
- Normalize widths/heights across panels
- Configurable spacing between panels

### Headers & Labels
- Primary and secondary row/column headers with full typography support
- Spanning headers across multiple columns/rows with bridging lines
- Swappable header positions (left/right, top/bottom)
- Per-character formatting: bold, italic, strikethrough, superscript, subscript
- Custom fonts (system fonts + uploaded fonts)
- Panel labels (a, b, c...) with draggable positioning

### Image Editing
- **Crop/Resize**: Aspect ratio locking (1:1, 4:3, 16:9, custom), rotation, flip
- **Adjustments**: Brightness, contrast, exposure, gamma, hue, saturation, vibrance, temperature, tint, sharpen, blur
- **Per-channel levels**: RGB histogram with input black/white controls
- Auto-adjust: levels, contrast, white balance
- Copy adjustments/crops across rows and columns

### Scale Bars
- Predefined and custom scale definitions (km, m, cm, mm, um, nm, pm)
- Auto unit conversion
- Customizable bar appearance (height, color, position)
- Full typography for scale bar labels
- Scale persists correctly across crop and resize operations

### Annotations
- **Symbols**: Arrow, star, rectangle, circle, cross, triangle, narrow triangle
- **Lines**: Straight (2-point) and multi-point with measurement
- **Areas**: Rectangle, ellipse, custom polygon, magic wand selection
- Smoothed vs discrete polygon modes
- Measurement text with full font customization and draggable positioning
- All annotations are crop-independent (positions preserved across crop changes)

### Zoomed Insets
- **Standard Zoom**: Overlay inset with connecting lines on the same panel
- **Adjacent Panel**: Zoomed content placed in a neighboring empty panel with cross-panel connecting lines
- **External Image**: Use a different image as the inset content (e.g., higher magnification view)
- Draggable source and target areas with resize handles

### Video Support
- Load video files (MP4, AVI, MOV, MKV, WebM, and more)
- Play & Seek tab with frame-by-frame navigation
- Play/pause with live preview
- Selected frame used for all image editing operations

### Save/Load
- **Save Project** (.mpf): Bundles all images, videos, fonts, and settings into a single shareable file
- **Load Project**: Full session restoration from .mpf file
- **Export Figure**: TIFF (lossless) or PNG at configurable DPI (72-600)
- Auto-generated filenames with date/time

### UI/UX
- Dark mode interface
- About page with citation and changelog
- In-app update checking
- Analysis panel with exportable measurement data (CSV)

## Installation

### Pre-built Releases
Download the latest release for your platform:
- **macOS (Apple Silicon)**: `.dmg` installer
- **macOS (Intel)**: `.dmg` installer
- **Windows**: `.msi` installer

### Build from Source

#### Prerequisites
- [Node.js](https://nodejs.org/) 18+
- [Rust](https://rustup.rs/) 1.77+
- [Python](https://python.org/) 3.10+
- [PyInstaller](https://pyinstaller.org/)

#### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/zhuojianlook/multipanelfigure.git
   cd multipanelfigure/TauriApp
   ```

2. **Install frontend dependencies**
   ```bash
   npm install
   ```

3. **Install Python dependencies**
   ```bash
   pip install pillow numpy matplotlib opencv-python-headless fastapi uvicorn
   ```

4. **Build the Python sidecar**
   ```bash
   chmod +x scripts/build-sidecar.sh
   ./scripts/build-sidecar.sh
   ```

5. **Run in development mode**
   ```bash
   # Start the Python sidecar
   cd python-sidecar && python api_server.py --port 8765 &
   cd ..

   # Start the Tauri app
   npx tauri dev
   ```

6. **Build for production**
   ```bash
   npx tauri build
   ```
   The installer will be in `src-tauri/target/release/bundle/`.

## Architecture

```
TauriApp/
  src/                    # React frontend (TypeScript + MUI)
  src-tauri/              # Tauri/Rust native shell
    binaries/             # Bundled Python sidecar (PyInstaller)
  python-sidecar/         # Python backend (FastAPI)
    api_server.py         # REST API server
    image_processing.py   # PIL/matplotlib image processing
    figure_builder.py     # Matplotlib figure assembly
    models.py             # Data models and serialization
    symbol_defs.py        # Annotation symbol definitions
  scripts/                # Build scripts
```

The app uses a client-server architecture:
- **Frontend**: React + MUI running in a Tauri WebView
- **Backend**: Python FastAPI server bundled as a standalone sidecar binary
- **Communication**: HTTP REST API on localhost

## Citation

If you use this tool in your research, please cite:

```
Look, Z. (2026). Multi-Panel Figure Builder [Version 0.1.0] [Computer software].
https://github.com/zhuojianlook/multipanelfigure
```

## License

MIT License

## Author

Created by **Zhuojian Look**
