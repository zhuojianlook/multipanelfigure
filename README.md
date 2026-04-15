# Multi-Panel Figure Builder

A professional desktop application for creating multi-panel scientific figures with full control over layout, annotations, scale bars, and image adjustments.

Built with Tauri (Rust + React) and a Python image processing backend.

## Download

Download the latest release for your platform from the [Releases page](https://github.com/zhuojianlook/multipanelfigure/releases/latest):

| Platform | File |
|---|---|
| macOS (Apple Silicon) | `.dmg` installer |
| Windows | `.exe` installer |

> **macOS users:** On first launch, right-click (or Control-click) the app and select **Open**. Click **Open** in the dialog to bypass Gatekeeper. This only needs to be done once.

## Features

- Configurable grid layout with drag-and-drop image assignment
- Primary and secondary row/column headers with rich text formatting
- Spanning headers across multiple columns/rows with bridging lines
- Image editing: crop, resize, brightness, contrast, levels, and more
- Scale bars with auto unit conversion and customizable appearance
- Annotations: arrows, shapes, lines, areas, and measurements
- Zoomed insets (standard, adjacent panel, external image)
- Video frame extraction (MP4, AVI, MOV, MKV, WebM)
- Save/load projects (.mpf), export as TIFF or PNG at configurable DPI

## Documentation

See the full documentation, build instructions, and architecture details in [TauriApp/README.md](TauriApp/README.md).

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
