/* ──────────────────────────────────────────────────────────
   VolumeViewer — Interactive 3D volume rendering of z-stack
   data using Three.js with ray-casting shader.
   ────────────────────────────────────────────────────────── */

import { useEffect, useRef, useState } from "react";
import {
  Dialog, DialogTitle, DialogContent, IconButton, Box, Typography,
  Slider, Select, MenuItem, Button, CircularProgress,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { api } from "../../api/client";

interface Props {
  open: boolean;
  onClose: () => void;
  imageName: string;
  startFrame: number;
  endFrame: number;
}

// ── GLSL Shaders for volume ray-casting ──────────────────

const vertexShader = `
varying vec3 vOrigin;
varying vec3 vDirection;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 cameraPos;

void main() {
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vOrigin = vec3(inverse(modelMatrix) * vec4(cameraPos, 1.0));
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * mvPosition;
}
`;

const fragmentShader = `
precision highp float;
precision highp sampler3D;

varying vec3 vOrigin;
varying vec3 vDirection;

uniform sampler3D volumeData;
uniform float threshold;
uniform float opacity;
uniform float steps;
uniform int colormap; // 0=gray, 1=hot, 2=cool, 3=viridis

vec3 applyColormap(float v, int cm) {
  if (cm == 1) { // hot
    return vec3(clamp(v * 3.0, 0.0, 1.0), clamp(v * 3.0 - 1.0, 0.0, 1.0), clamp(v * 3.0 - 2.0, 0.0, 1.0));
  } else if (cm == 2) { // cool
    return vec3(v, 1.0 - v, 1.0);
  } else if (cm == 3) { // viridis-like
    return vec3(0.267 + v * 0.329, 0.004 + v * 0.873, 0.329 + v * (0.893 - 0.329) * (1.0 - v * 0.5));
  }
  return vec3(v); // grayscale
}

vec2 intersectBox(vec3 orig, vec3 dir) {
  vec3 invDir = 1.0 / dir;
  vec3 tMin = (vec3(0.0) - orig) * invDir;
  vec3 tMax = (vec3(1.0) - orig) * invDir;
  vec3 t1 = min(tMin, tMax);
  vec3 t2 = max(tMin, tMax);
  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar = min(min(t2.x, t2.y), t2.z);
  return vec2(tNear, tFar);
}

void main() {
  vec3 rayDir = normalize(vDirection);
  vec2 bounds = intersectBox(vOrigin, rayDir);

  if (bounds.x > bounds.y) discard;

  bounds.x = max(bounds.x, 0.0);
  float stepSize = 1.0 / steps;
  vec3 pos = vOrigin + bounds.x * rayDir;
  vec3 step = rayDir * stepSize;

  vec4 color = vec4(0.0);

  for (float t = bounds.x; t < bounds.y; t += stepSize) {
    float val = texture(volumeData, pos).r;
    if (val > threshold) {
      vec3 c = applyColormap(val, colormap);
      float a = val * opacity;
      color.rgb += (1.0 - color.a) * a * c;
      color.a += (1.0 - color.a) * a;
      if (color.a > 0.95) break;
    }
    pos += step;
  }

  gl_FragColor = color;
}
`;

export function VolumeViewerDialog({ open, onClose, imageName, startFrame, endFrame }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [thresholdVal, setThresholdVal] = useState(0.1);
  const [opacityVal, setOpacityVal] = useState(0.5);
  const [zSpacing, setZSpacing] = useState(1.0);
  const [colormap, setColormap] = useState(0);
  const uniformsRef = useRef<Record<string, THREE.IUniform>>({});
  const meshRef = useRef<THREE.Mesh | null>(null);

  useEffect(() => {
    if (!open || !canvasRef.current) return;

    let disposed = false;
    const canvas = canvasRef.current;

    const init = async () => {
      setLoading(true);
      setError("");
      try {
        const vol = await api.getVolumeData(imageName, startFrame, endFrame, 128);

        if (disposed) return;

        // Decode base64 volume data
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

        // Setup Three.js scene
        const w = canvas.clientWidth || 600;
        const h = canvas.clientHeight || 500;
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        renderer.setSize(w, h);
        renderer.setPixelRatio(window.devicePixelRatio);
        rendererRef.current = renderer;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1c1c1e);

        const camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 10);
        camera.position.set(1.5, 1.5, 1.5);
        camera.lookAt(0.5, 0.5, 0.5);

        const controls = new OrbitControls(camera, canvas);
        controls.target.set(0.5, 0.5, 0.5);
        controls.update();

        // Volume box geometry (unit cube)
        const geometry = new THREE.BoxGeometry(1, 1, 1);

        const uniforms: Record<string, THREE.IUniform> = {
          volumeData: { value: texture },
          threshold: { value: thresholdVal },
          opacity: { value: opacityVal },
          steps: { value: 200.0 },
          colormap: { value: colormap },
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
out vec3 vOrigin;
out vec3 vDirection;
void main() {
  vOrigin = vec3(inverse(modelMatrix) * vec4(cameraPos, 1.0));
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`,
          fragmentShader: `#version 300 es
precision highp float;
precision highp sampler3D;
in vec3 vOrigin;
in vec3 vDirection;
out vec4 fragColor;
uniform sampler3D volumeData;
uniform float threshold;
uniform float opacity;
uniform float steps;
uniform int colormap;

vec3 applyColormap(float v, int cm) {
  if (cm == 1) return vec3(clamp(v*3.0,0.0,1.0), clamp(v*3.0-1.0,0.0,1.0), clamp(v*3.0-2.0,0.0,1.0));
  if (cm == 2) return vec3(v, 1.0-v, 1.0);
  if (cm == 3) return vec3(0.27+v*0.33, v*0.87, 0.33+v*0.56*(1.0-v*0.5));
  return vec3(v);
}

vec2 intersectBox(vec3 orig, vec3 dir) {
  vec3 inv = 1.0/dir;
  vec3 t1 = min(-orig*inv, (vec3(1.0)-orig)*inv);
  vec3 t2 = max(-orig*inv, (vec3(1.0)-orig)*inv);
  return vec2(max(max(t1.x,t1.y),t1.z), min(min(t2.x,t2.y),t2.z));
}

void main() {
  vec3 rd = normalize(vDirection);
  vec2 b = intersectBox(vOrigin, rd);
  if (b.x > b.y) discard;
  b.x = max(b.x, 0.0);
  float ss = 1.0/steps;
  vec3 p = vOrigin + b.x*rd;
  vec4 col = vec4(0.0);
  for (float t=b.x; t<b.y; t+=ss) {
    float v = texture(volumeData, p).r;
    if (v > threshold) {
      vec3 c = applyColormap(v, colormap);
      float a = v * opacity;
      col.rgb += (1.0-col.a)*a*c;
      col.a += (1.0-col.a)*a;
      if (col.a > 0.95) break;
    }
    p += rd*ss;
  }
  fragColor = col;
}`,
          side: THREE.BackSide,
          transparent: true,
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.scale.set(1, 1, zSpacing);
        scene.add(mesh);
        meshRef.current = mesh;

        // Render loop
        const animate = () => {
          if (disposed) return;
          requestAnimationFrame(animate);
          uniforms.cameraPos.value.copy(camera.position);
          controls.update();
          renderer.render(scene, camera);
        };
        animate();

        // Handle resize
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
    if (uniformsRef.current.threshold) uniformsRef.current.threshold.value = thresholdVal;
    if (uniformsRef.current.opacity) uniformsRef.current.opacity.value = opacityVal;
    if (uniformsRef.current.colormap) uniformsRef.current.colormap.value = colormap;
    if (meshRef.current) meshRef.current.scale.z = zSpacing;
  }, [thresholdVal, opacityVal, colormap, zSpacing]);

  const saveView = () => {
    if (!canvasRef.current) return;
    const link = document.createElement("a");
    link.href = canvasRef.current.toDataURL("image/png");
    link.download = `volume_${imageName.replace(/\.\w+$/, "")}.png`;
    link.click();
  };

  return (
    <Dialog open={open} onClose={onClose} fullScreen>
      <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", py: 1, px: 2 }}>
        <Typography variant="h6" sx={{ fontSize: "1rem", fontWeight: 700 }}>3D Volume View — {imageName}</Typography>
        <IconButton onClick={onClose} size="small"><CloseIcon /></IconButton>
      </DialogTitle>
      <DialogContent sx={{ p: 0, display: "flex", height: "100%", overflow: "hidden" }}>
        {/* Canvas */}
        <Box sx={{ flex: 1, position: "relative" }}>
          <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
          {loading && (
            <Box sx={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "rgba(0,0,0,0.6)" }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <CircularProgress size={20} />
                <Typography variant="caption">Loading volume data...</Typography>
              </Box>
            </Box>
          )}
          {error && (
            <Box sx={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "rgba(0,0,0,0.6)" }}>
              <Typography color="error" variant="caption">{error}</Typography>
            </Box>
          )}
        </Box>

        {/* Controls panel */}
        <Box sx={{ width: 220, flexShrink: 0, borderLeft: 1, borderColor: "divider", p: 2, display: "flex", flexDirection: "column", gap: 2, overflow: "auto" }}>
          <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>Controls</Typography>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Threshold</Typography>
            <Slider size="small" value={thresholdVal} min={0} max={1} step={0.01}
              onChange={(_, v) => setThresholdVal(v as number)} />
          </Box>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Opacity</Typography>
            <Slider size="small" value={opacityVal} min={0.01} max={2} step={0.01}
              onChange={(_, v) => setOpacityVal(v as number)} />
          </Box>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Z Spacing</Typography>
            <Slider size="small" value={zSpacing} min={0.1} max={5} step={0.1}
              onChange={(_, v) => setZSpacing(v as number)} />
          </Box>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Colormap</Typography>
            <Select size="small" value={colormap} onChange={(e) => setColormap(Number(e.target.value))}
              sx={{ fontSize: "0.65rem", width: "100%", "& .MuiSelect-select": { py: 0.3 } }}>
              <MenuItem value={0} sx={{ fontSize: "0.65rem" }}>Grayscale</MenuItem>
              <MenuItem value={1} sx={{ fontSize: "0.65rem" }}>Hot</MenuItem>
              <MenuItem value={2} sx={{ fontSize: "0.65rem" }}>Cool</MenuItem>
              <MenuItem value={3} sx={{ fontSize: "0.65rem" }}>Viridis</MenuItem>
            </Select>
          </Box>

          <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary" }}>
            Drag to rotate. Scroll to zoom. Right-click to pan.
          </Typography>

          <Button size="small" variant="outlined" onClick={saveView} sx={{ fontSize: "0.6rem", textTransform: "none" }}>
            Save View as PNG
          </Button>
        </Box>
      </DialogContent>
    </Dialog>
  );
}
