# Cherry‑Maturity YOLOv11

Detecta y clasifica el grado de madurez de cerezas con **YOLOv11** usando un contenedor **Singularity/Apptainer** para máxima portabilidad.

---

## 🗂️ Estructura del repo

```text
cherry-maturity-yolo/
│
├── README.md               ← este archivo
├── LICENSE
├── .gitignore              ← ignora data/, outputs/, *.sif, __pycache__ …
│
├── singularity/
│   ├── yolo11.def          ← receta del contenedor
│   └── build_container.sh  ← wrapper: construye yolo11.sif
│
├── configs/
│   └── cherries_maturity.yaml
│
├── requirements.txt        ← dependencias Python (cu121)
│
├── scripts/
│   ├── train.py            ← script principal de entrenamiento
│   ├── submit_slurm.sh     ← plantilla para Slurm
│   └── download_dataset.sh ← descarga y prepara el dataset
│
├── src/                    ← (utilidades opcionales)
│   └── utils.py
│
├── data/                   ← vacío; se llena con el dataset (no se versiona)
└── outputs/                ← modelos, logs, métricas (no se versiona)
```

---

## ⚙️ Requisitos

| Recurso                     | Versión mínima                                         |
| --------------------------- | ------------------------------------------------------ |
| **Singularity / Apptainer** | 3.11                                                   |
| **GPU driver**              |  NVIDIA ≥ 546.xx (CUDA 12.8)                           |
| **Almacenamiento**          |  ≈ 30 GB (dataset + modelos)                           |
| **RAM**                     |  ≥ 32 GB                                               |
| **Python host**             |  3.10 (para scripts auxiliares, no para entrenamiento) |

> **Nota:** El contenedor incluye PyTorch 2.4.1 + cu121. Un driver 12.8 ejecuta binarios cu121 sin problemas.

---

## 🚀 Uso rápido

```bash
# 1) Clona el repo
$ git clone git@github.com:garcesfruit-data/cherry-maturity-yolo.git
$ cd cherry-maturity-yolo

# 2) Construye el contenedor (≈ 15 min)
$ ./singularity/build_container.sh      # genera singularity/yolo11.sif

# 3) Descarga el dataset (~8 GB)
$ ./scripts/download_dataset.sh

# 4) Entrena en local (3 GPUs)
$ singularity exec --nv singularity/yolo11.sif \
      python3 scripts/train.py \
      --config configs/cherries_maturity.yaml

# 5) Entrena en Slurm
$ sbatch scripts/submit_slurm.sh
```

### Parámetros clave (`train.py`)

```text
--epochs          # default: 120
--img-sizes       # lista de resoluciones, ej. 1024 640
--batches         # lista de tamaños de batch
--device          # 'auto', '0,1,2', etc.
--project         # carpeta raíz de salidas
```

Ejemplo:

```bash
singularity exec --nv singularity/yolo11.sif \
  python3 scripts/train.py \
  --config configs/cherries_maturity.yaml \
  --epochs 150 --img-sizes 1280 1024 \
  --batches 64 32 16 --device 0,1
```

---

## 📂 Estructura de salidas

```
outputs/
└── cherry_yolo11_model/
    ├── sz1024_bs64/        ← run_name
    │   ├── model.pt        ← modelo final
    │   └── metrics.csv     ← métricas por época
    ├── sz1024_bs32/ …
    └── training_summary.csv
```

---

## 🛠️ Solución de problemas

| Síntoma                                             | Causa probable                       | Fix                                                                             |
| --------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| `CUDA driver too old`                               | Driver < 546.xx                      | Actualiza el driver o usa nodo compatible                                       |
| `No module named ultralytics` dentro del contenedor | Falló la instalación de requirements | Re‑construye el `.sif` o instala manualmente `pip install ultralytics==8.3.154` |
| GPU 0 out‑of‑memory                                 | **batch/img‑size** muy altos         | Reduce `--batches` o usa más GPUs                                               |

---

## 📝 Licencia

Código bajo **MIT**. El dataset se distribuye sólo para investigación interna; revisa su licencia antes de redistribuir.

---

## 🤝 Créditos

Desarrollo original: **Fabián Escobar** & **Camilo Aliste** (2025). Inspirado en Ultralytics YOLO.

¡Feliz cosecha de cerezas 🍒!

