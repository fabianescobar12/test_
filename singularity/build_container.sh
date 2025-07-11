#!/usr/bin/env bash
set -e
DEF=singularity/yolo11.def
SIF=singularity/yolo11.sif

echo "▶️ Construyendo $SIF …"
singularity build "$SIF" "$DEF"
echo "✅  Contenedor listo: $SIF"
