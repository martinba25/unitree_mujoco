#!/usr/bin/env bash
# deploy_g1.sh
# Clara → Paperspace: sync, render, retrieve
# Clara ist die einzige Quelle der Wahrheit.

set -euo pipefail

REMOTE="paperspace"
REMOTE_DIR="~/unitree_mujoco"
LOCAL_DIR="/home/martinba/unitree_mujoco"
SCENE="retail/scene_sorting_v2.xml"
CAMERA="pitch_cam"
OUTPUT_PNG="retail/demo/G1D_Layout_Final.png"

echo "═══════════════════════════════════════════════"
echo " deploy_g1.sh  |  Clara → Paperspace"
echo "═══════════════════════════════════════════════"

# ── 1. rsync: alle .xml und .py Dateien, ohne .git und Bilder ──────────
echo ""
echo "▶ [1/4] rsync XML + Python → ${REMOTE}:${REMOTE_DIR}/"
rsync -avz --checksum \
    --include="*/" \
    --include="*.xml" \
    --include="*.py" \
    --include="*.stl" \
    --include="*.STL" \
    --include="*.obj" \
    --include="*.mtl" \
    --exclude="*.png" \
    --exclude="*.jpg" \
    --exclude="*.jpeg" \
    --exclude="*.gif" \
    --exclude="*.bmp" \
    --exclude="*.tiff" \
    --exclude="*.svg" \
    --exclude=".git/" \
    --exclude="*.pyc" \
    --exclude="__pycache__/" \
    --exclude="*.egg-info/" \
    --exclude=".env" \
    --exclude="*" \
    "${LOCAL_DIR}/" \
    "${REMOTE}:${REMOTE_DIR}/"
echo "   rsync abgeschlossen."

# ── 2. Mesh-Symlinks + retail/demo/ sicherstellen ──────────────────────
echo ""
echo "▶ [2/4] Mesh-Symlinks und retail/demo/ anlegen"
ssh "${REMOTE}" "
  mkdir -p ${REMOTE_DIR}/retail/demo
  cd ${REMOTE_DIR}/unitree_robots/g1
  for f in meshes/*.STL; do ln -sf \"\$f\" . 2>/dev/null || true; done
  echo '   Mesh-Symlinks OK'
"

# ── 3. MuJoCo Render auf Paperspace ────────────────────────────────────
echo ""
echo "▶ [3/4] MuJoCo render auf Paperspace: ${SCENE} | Kamera: ${CAMERA}"
ssh "${REMOTE}" "MUJOCO_GL=egl python3 - <<'PYEOF'
import mujoco
import imageio
import numpy as np
import os

base = os.path.expanduser('~/unitree_mujoco')
xml_path    = os.path.join(base, 'retail/scene_sorting_v2.xml')
output_path = os.path.join(base, 'retail/demo/G1D_Layout_Final.png')

print(f'  Lade Modell: {xml_path}')
model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

# Kamera-ID suchen
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'pitch_cam')
if cam_id < 0:
    raise RuntimeError('Kamera pitch_cam nicht gefunden!')
print(f'  Kamera pitch_cam ID: {cam_id}')

# Physik kurz einlaufen lassen damit Objekte liegen
mujoco.mj_forward(model, data)

# Offscreen rendern (kein Display erforderlich)
renderer = mujoco.Renderer(model, height=1080, width=1920)
renderer.update_scene(data, camera='pitch_cam')
pixels = renderer.render()

os.makedirs(os.path.dirname(output_path), exist_ok=True)
imageio.imwrite(output_path, pixels)
print(f'  Bild gespeichert: {output_path}')
print(f'  Auflösung: {pixels.shape[1]}x{pixels.shape[0]}')
PYEOF"

# ── 4. Bild zurück nach Clara kopieren ──────────────────────────────────
echo ""
echo "▶ [4/4] Bild von Paperspace → Clara: retail/demo/"
mkdir -p "${LOCAL_DIR}/retail/demo"
scp "${REMOTE}:${REMOTE_DIR}/${OUTPUT_PNG}" \
    "${LOCAL_DIR}/retail/demo/G1D_Layout_Final.png"

echo ""
echo "═══════════════════════════════════════════════"
echo " Fertig. Bild gespeichert:"
echo "   ${LOCAL_DIR}/retail/demo/G1D_Layout_Final.png"
echo "═══════════════════════════════════════════════"
