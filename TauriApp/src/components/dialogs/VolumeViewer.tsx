/* ──────────────────────────────────────────────────────────
   VolumeViewer — Hybrid 3D volume rendering.
   Client-side Three.js for real-time interaction (smooth rotation/zoom).
   Server-side for high-res save/use-as-panel export.
   ────────────────────────────────────────────────────────── */

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Dialog, DialogTitle, DialogContent, DialogActions, IconButton, Box, Typography,
  Slider, Select, MenuItem, Button, CircularProgress, ToggleButtonGroup, ToggleButton,
  FormControlLabel, Checkbox, TextField,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import SaveAltIcon from "@mui/icons-material/SaveAlt";
import ImageIcon from "@mui/icons-material/Image";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { api } from "../../api/client";
import { useFigureStore } from "../../store/figureStore";

interface Props {
  open: boolean;
  onClose: () => void;
  imageName: string;
  startFrame: number;
  endFrame: number;
  panelRow?: number;
  panelCol?: number;
}

// Colormap shader helper (GLSL)
const colormapGLSL = `
vec3 applyColormap(float v, int cm) {
  if (cm == 1) return vec3(clamp(v*3.0,0.0,1.0), clamp(v*3.0-1.0,0.0,1.0), clamp(v*3.0-2.0,0.0,1.0));
  if (cm == 2) return vec3(v, 1.0-v, 1.0);
  if (cm == 3) return vec3(0.267+v*0.062, 0.005+v*0.897, 0.329+v*0.266);
  if (cm == 4) return vec3(v*0.8, v*v*0.5, v*0.3+0.1);
  if (cm == 5) return vec3(v, v*v*0.3, v*v*0.1);
  if (cm == 6) return vec3(0.05+v*0.89, 0.03+v*0.74, v*v*0.6);
  return vec3(v);
}
`;

const CMAP_NAMES = ["gray", "hot", "cool", "viridis", "magma", "inferno", "plasma"];

export function VolumeViewerDialog({ open, onClose, imageName, startFrame, endFrame, panelRow, panelCol }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const uniformsRef = useRef<Record<string, THREE.IUniform>>({});
  const meshRef = useRef<THREE.Mesh | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [threshold, setThreshold] = useState(0.2);
  const [opacity, setOpacity] = useState(0.6);
  const [zSpacing, setZSpacing] = useState(1.0);
  const [colormap, setColormap] = useState(0);
  const [showAxes, setShowAxes] = useState(true);

  // Save dialog
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveFormat, setSaveFormat] = useState("PNG");
  const [saveQuality, setSaveQuality] = useState(95);
  const [savePath, setSavePath] = useState("");
  const [saving, setSaving] = useState(false);
  const fetchImages = useFigureStore((s) => s.fetchImages);

  // Initialize Three.js scene with volume data
  useEffect(() => {
    if (!open || !canvasRef.current) return;
    let disposed = false;
    const canvas = canvasRef.current;

    const init = async () => {
      setLoading(true);
      setError("");
      try {
        // Fetch small volume for fast interaction (64³ max)
        const vol = await api.getVolumeData(imageName, startFrame, endFrame, 96);
        if (disposed) return;

        // Decode base64 → Uint8Array
        const raw = atob(vol.data);
        const data = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) data[i] = raw.charCodeAt(i);

        // Create 3D texture
        const texture = new THREE.Data3DTexture(data, vol.width, vol.height, vol.depth);
        texture.format = THREE.RedFormat;
        texture.type = THREE.UnsignedByteType;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.needsUpdate = true;

        // Setup Three.js
        const w = canvas.clientWidth || 900;
        const h = canvas.clientHeight || 700;
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        renderer.setSize(w, h);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        rendererRef.current = renderer;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1c1c1e);
        sceneRef.current = scene;

        const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
        const maxDim = Math.max(vol.width, vol.height, vol.depth * zSpacing);
        camera.position.set(maxDim * 1.5, maxDim * 1.5, maxDim * 1.5);
        camera.lookAt(vol.width / 2, vol.height / 2, (vol.depth * zSpacing) / 2);
        cameraRef.current = camera;

        const controls = new OrbitControls(camera, canvas);
        controls.target.set(vol.width / 2, vol.height / 2, (vol.depth * zSpacing) / 2);
        controls.enableDamping = true;
        controls.dampingFactor = 0.1;
        controls.update();
        controlsRef.current = controls;

        // Volume box
        const geometry = new THREE.BoxGeometry(vol.width, vol.height, vol.depth * zSpacing);
        geometry.translate(vol.width / 2, vol.height / 2, (vol.depth * zSpacing) / 2);

        const uniforms: Record<string, THREE.IUniform> = {
          volumeData: { value: texture },
          threshold: { value: threshold },
          opacity: { value: opacity },
          steps: { value: 100.0 },
          colormap: { value: colormap },
          volumeSize: { value: new THREE.Vector3(vol.width, vol.height, vol.depth * zSpacing) },
          cameraPos: { value: camera.position },
        };
        uniformsRef.current = uniforms;

        const material = new THREE.RawShaderMaterial({
          glslVersion: THREE.GLSL3,
          uniforms,
          vertexShader: `#version 300 es
in vec3 position;
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 cameraPos;
uniform vec3 volumeSize;
out vec3 vOrigin;
out vec3 vDirection;
out vec3 vLocal;
void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vLocal = position / volumeSize;
  vOrigin = (inverse(modelMatrix) * vec4(cameraPos, 1.0)).xyz / volumeSize;
  vDirection = (position / volumeSize) - vOrigin;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`,
          fragmentShader: `#version 300 es
precision highp float;
precision highp sampler3D;
in vec3 vOrigin;
in vec3 vDirection;
in vec3 vLocal;
out vec4 fragColor;
uniform sampler3D volumeData;
uniform float threshold;
uniform float opacity;
uniform float steps;
uniform int colormap;

${colormapGLSL}

vec2 intersectBox(vec3 orig, vec3 dir) {
  vec3 inv = 1.0 / dir;
  vec3 t1 = (vec3(0.0) - orig) * inv;
  vec3 t2 = (vec3(1.0) - orig) * inv;
  vec3 tMin = min(t1, t2);
  vec3 tMax = max(t1, t2);
  return vec2(max(max(tMin.x, tMin.y), tMin.z), min(min(tMax.x, tMax.y), tMax.z));
}

void main() {
  vec3 rd = normalize(vDirection);
  vec2 b = intersectBox(vOrigin, rd);
  if (b.x > b.y) discard;
  b.x = max(b.x, 0.0);
  float stepSize = 1.732 / steps;
  vec3 pos = vOrigin + b.x * rd;
  vec3 step = rd * stepSize;
  vec4 col = vec4(0.0);
  for (float t = b.x; t < b.y; t += stepSize) {
    float v = texture(volumeData, pos).r;
    if (v > threshold) {
      vec3 c = applyColormap(v, colormap);
      float a = v * opacity;
      col.rgb += (1.0 - col.a) * a * c;
      col.a += (1.0 - col.a) * a;
      if (col.a > 0.95) break;
    }
    pos += step;
  }
  if (col.a < 0.01) discard;
  fragColor = col;
}`,
          side: THREE.BackSide,
          transparent: true,
        });

        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
        meshRef.current = mesh;

        // Axes helper
        if (showAxes) {
          const axesHelper = new THREE.AxesHelper(Math.max(vol.width, vol.height, vol.depth) * 1.1);
          axesHelper.name = "axes";
          scene.add(axesHelper);
          const gridHelper = new THREE.GridHelper(Math.max(vol.width, vol.height) * 1.5, 10, 0x666666, 0x333333);
          gridHelper.name = "grid";
          scene.add(gridHelper);
        }

        // Render loop
        const animate = () => {
          if (disposed) return;
          requestAnimationFrame(animate);
          uniforms.cameraPos.value.copy(camera.position);
          controls.update();
          renderer.render(scene, camera);
        };
        animate();

        // Resize handler
        const onResize = () => {
          const w2 = canvas.clientWidth;
          const h2 = canvas.clientHeight;
          if (w2 > 0 && h2 > 0) {
            camera.aspect = w2 / h2;
            camera.updateProjectionMatrix();
            renderer.setSize(w2, h2);
          }
        };
        window.addEventListener("resize", onResize);

        setLoading(false);

        return () => {
          disposed = true;
          window.removeEventListener("resize", onResize);
          controls.dispose();
          renderer.dispose();
          geometry.dispose();
          material.dispose();
          texture.dispose();
        };
      } catch (e) {
        if (!disposed) {
          setError(e instanceof Error ? e.message : String(e));
          setLoading(false);
        }
      }
    };

    const cleanup = init();
    return () => { disposed = true; cleanup?.then(fn => fn?.()); };
  }, [open, imageName, startFrame, endFrame]); // eslint-disable-line

  // Update uniforms when sliders change
  useEffect(() => {
    if (uniformsRef.current.threshold) uniformsRef.current.threshold.value = threshold;
    if (uniformsRef.current.opacity) uniformsRef.current.opacity.value = opacity;
    if (uniformsRef.current.colormap) uniformsRef.current.colormap.value = colormap;
  }, [threshold, opacity, colormap]);

  // Update z-spacing
  useEffect(() => {
    if (!meshRef.current || !uniformsRef.current.volumeSize) return;
    const vs = uniformsRef.current.volumeSize.value as THREE.Vector3;
    const newDepth = (vs.z / (meshRef.current.scale.z || 1)) * zSpacing;
    // Simpler: just scale the mesh
    meshRef.current.scale.z = zSpacing;
  }, [zSpacing]);

  // Show/hide axes
  useEffect(() => {
    if (!sceneRef.current) return;
    const axes = sceneRef.current.getObjectByName("axes");
    const grid = sceneRef.current.getObjectByName("grid");
    if (axes) axes.visible = showAxes;
    if (grid) grid.visible = showAxes;
  }, [showAxes]);

  const getCurrentViewParams = () => ({
    startFrame, endFrame,
    elev: 30, azim: -60, // will be replaced by camera-derived values
    threshold, zSpacing,
    colormap: CMAP_NAMES[colormap] ?? "gray",
    showAxes, zoom: 1.0,
  });

  const saveViewAsPng = () => {
    if (!canvasRef.current || !rendererRef.current || !sceneRef.current || !cameraRef.current) return;
    rendererRef.current.render(sceneRef.current, cameraRef.current);
    const link = document.createElement("a");
    link.href = canvasRef.current.toDataURL("image/png");
    link.download = `volume_${imageName.replace(/\.\w+$/, "")}.png`;
    link.click();
  };

  const openSaveDialog = async () => {
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const ext = saveFormat === "TIFF" ? "tiff" : saveFormat.toLowerCase();
      const path = await save({
        defaultPath: `volume_${imageName.replace(/\.\w+$/, "")}.${ext}`,
        filters: [{ name: saveFormat, extensions: [ext] }],
      });
      if (path) {
        setSavePath(path as string);
        setSaveOpen(true);
      } else {
        // Fallback: save current canvas as PNG
        saveViewAsPng();
      }
    } catch {
      saveViewAsPng();
    }
  };

  const performSave = async () => {
    if (!savePath) return;
    setSaving(true);
    try {
      if (saveFormat === "PNG" && canvasRef.current) {
        // Use canvas for PNG (client-side, instant)
        const dataUrl = canvasRef.current.toDataURL("image/png");
        const bin = atob(dataUrl.split(",")[1]);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        // Write via Rust command or fallback to download
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          const b64 = btoa(String.fromCharCode(...bytes));
          await invoke("save_base64_to_path", { path: savePath, dataB64: b64 });
        } catch {
          const link = document.createElement("a");
          link.href = dataUrl;
          link.download = savePath.split("/").pop() || "volume.png";
          link.click();
        }
      } else {
        // Use server-side rendering for TIFF/JPEG (client canvas doesn't support them well)
        await api.saveVolumeRenderAsImage(imageName, {
          ...getCurrentViewParams(),
          width: 1600, height: 1200,
          format: saveFormat, quality: saveQuality, filePath: savePath,
        });
      }
      setSaveOpen(false);
    } catch (e) {
      setError("Save failed: " + (e instanceof Error ? e.message : String(e)));
    }
    setSaving(false);
  };

  const useAsPanel = async () => {
    if (panelRow == null || panelCol == null) return;
    try {
      // Capture current canvas as PNG, use as panel via server
      await api.useVolumeAsPanel(imageName, panelRow, panelCol, {
        ...getCurrentViewParams(),
        method: "surface", width: 1600, height: 1200,
      });
      await fetchImages();
      onClose();
    } catch (e) {
      setError("Failed to set as panel: " + (e instanceof Error ? e.message : String(e)));
    }
  };

  return (
    <>
    <Dialog open={open} onClose={onClose} fullScreen>
      <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", py: 1, px: 2 }}>
        <Typography variant="h6" sx={{ fontSize: "1rem", fontWeight: 700 }}>3D Volume View — {imageName}</Typography>
        <IconButton onClick={onClose} size="small"><CloseIcon /></IconButton>
      </DialogTitle>
      <DialogContent sx={{ p: 0, display: "flex", height: "100%", overflow: "hidden" }}>
        <Box sx={{ flex: 1, position: "relative" }}>
          <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
          {loading && (
            <Box sx={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "rgba(0,0,0,0.5)" }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, bgcolor: "background.paper", px: 2, py: 1, borderRadius: 1 }}>
                <CircularProgress size={16} />
                <Typography variant="caption">Loading volume...</Typography>
              </Box>
            </Box>
          )}
          {error && (
            <Box sx={{ position: "absolute", bottom: 16, left: "50%", transform: "translateX(-50%)", bgcolor: "error.dark", color: "white", px: 2, py: 1, borderRadius: 1 }}>
              <Typography variant="caption">{error}</Typography>
            </Box>
          )}
        </Box>

        <Box sx={{ width: 240, flexShrink: 0, borderLeft: 1, borderColor: "divider", p: 2, display: "flex", flexDirection: "column", gap: 1.5, overflow: "auto" }}>
          <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>Controls</Typography>

          <FormControlLabel
            sx={{ ml: 0 }}
            control={<Checkbox size="small" checked={showAxes} onChange={(e) => setShowAxes(e.target.checked)} sx={{ p: 0.25 }} />}
            label={<Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Show axes & grid</Typography>}
          />

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Threshold: {threshold.toFixed(2)}</Typography>
            <Slider size="small" value={threshold} min={0} max={1} step={0.01} onChange={(_, v) => setThreshold(v as number)} />
          </Box>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Opacity: {opacity.toFixed(2)}</Typography>
            <Slider size="small" value={opacity} min={0.01} max={2} step={0.01} onChange={(_, v) => setOpacity(v as number)} />
          </Box>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Z Spacing: {zSpacing.toFixed(1)}</Typography>
            <Slider size="small" value={zSpacing} min={0.1} max={5} step={0.1} onChange={(_, v) => setZSpacing(v as number)} />
          </Box>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Colormap</Typography>
            <Select size="small" value={colormap} onChange={(e) => setColormap(Number(e.target.value))}
              sx={{ fontSize: "0.65rem", width: "100%", "& .MuiSelect-select": { py: 0.3 } }}>
              <MenuItem value={0} sx={{ fontSize: "0.65rem" }}>Grayscale</MenuItem>
              <MenuItem value={1} sx={{ fontSize: "0.65rem" }}>Hot</MenuItem>
              <MenuItem value={2} sx={{ fontSize: "0.65rem" }}>Cool</MenuItem>
              <MenuItem value={3} sx={{ fontSize: "0.65rem" }}>Viridis</MenuItem>
              <MenuItem value={4} sx={{ fontSize: "0.65rem" }}>Magma</MenuItem>
              <MenuItem value={5} sx={{ fontSize: "0.65rem" }}>Inferno</MenuItem>
              <MenuItem value={6} sx={{ fontSize: "0.65rem" }}>Plasma</MenuItem>
            </Select>
          </Box>

          <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.secondary" }}>
            Drag to rotate. Scroll to zoom. Right-click to pan.
          </Typography>

          <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, mt: 1 }}>
            <Button size="small" variant="outlined" onClick={openSaveDialog} startIcon={<SaveAltIcon sx={{ fontSize: 12 }} />}
              sx={{ fontSize: "0.6rem", textTransform: "none" }}>
              Save View...
            </Button>
            {panelRow != null && panelCol != null && (
              <Button size="small" variant="contained" color="primary" onClick={useAsPanel} startIcon={<ImageIcon sx={{ fontSize: 12 }} />}
                sx={{ fontSize: "0.6rem", textTransform: "none" }}>
                Use as Panel Image
              </Button>
            )}
          </Box>
        </Box>
      </DialogContent>
    </Dialog>

    <Dialog open={saveOpen} onClose={() => setSaveOpen(false)} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ fontSize: "1rem" }}>Save Volume View</DialogTitle>
      <DialogContent>
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2, pt: 1 }}>
          <TextField label="File path" size="small" fullWidth value={savePath} onChange={(e) => setSavePath(e.target.value)} />
          <Box>
            <Typography variant="caption">Format</Typography>
            <ToggleButtonGroup value={saveFormat} exclusive onChange={(_, v) => { if (v) setSaveFormat(v); }} size="small" sx={{ mt: 0.5 }}>
              <ToggleButton value="PNG">PNG</ToggleButton>
              <ToggleButton value="TIFF">TIFF</ToggleButton>
              <ToggleButton value="JPEG">JPEG</ToggleButton>
            </ToggleButtonGroup>
          </Box>
          {saveFormat === "JPEG" && (
            <Box>
              <Typography variant="caption">Quality: {saveQuality}</Typography>
              <Slider size="small" value={saveQuality} min={1} max={100} step={1} onChange={(_, v) => setSaveQuality(v as number)} />
            </Box>
          )}
          <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.65rem" }}>
            {saveFormat === "PNG" ? "PNG uses the current live view (instant)." : "TIFF/JPEG render via backend at 1600x1200."}
          </Typography>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setSaveOpen(false)}>Cancel</Button>
        <Button variant="contained" onClick={performSave} disabled={saving || !savePath}>
          {saving ? "Saving..." : "Save"}
        </Button>
      </DialogActions>
    </Dialog>
    </>
  );
}
