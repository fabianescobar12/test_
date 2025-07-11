"""
Entrenamiento YOLOv11/paralelo en 3 GPUs
Fabián Escobar, Camilo Aliste – 2025
"""

from ultralytics import YOLO
import torch, pandas as pd
import os, time, datetime

ROOT_DIR      = "/mnt/beegfs/home/fescobar/Entrenamiento"
CFG_FILE      = f"{ROOT_DIR}/configs/cherries_maturity.yaml"
MODEL_WEIGHTS = "yolo11s.pt"
SUMMARY_FILE  = os.path.join(ROOT_DIR, "training_summary.csv")


DEVICES   = "0,1,2"          
WORLD_SIZE = len(DEVICES.split(","))

img_sizes = [1024, 640]
batch_map = {
    1024: [102, 69, 33, 18,  9],
    640 : [102, 69, 33, 18,  9]
}

def main() -> None:
    results_log = []

    for img_size in img_sizes:
        for batch_size in batch_map[img_size]:

            run_name  = f"sz{img_size}_bs{batch_size}"
            model_dir = os.path.join(ROOT_DIR, run_name)
            os.makedirs(model_dir, exist_ok=True)

            model = YOLO(MODEL_WEIGHTS)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            model.train(
                data      = CFG_FILE,
                epochs    = 120,
                imgsz     = img_size,
                batch     = batch_size, 
                fraction  = 1.0,
                project   = "cherry_yolo11_model",
                name      = run_name,
                device    = DEVICES,   
                workers   = 8,        
            )
            t_train = time.time() - t0

            val_res = model.val(
                data  = CFG_FILE,
                imgsz = img_size,
                batch = batch_size,
                device = 0            
            )

            perf_map50 = (
                float(val_res.box.map50)
                if hasattr(val_res, "box") else
                val_res.results_dict.get("metrics/mAP50(B)", None)
            )

            vram_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

            model_file = f"model_{batch_size}_{img_size}.pt"
            model.save(os.path.join(model_dir, model_file))
            print(f"✅  [{run_name}] Modelo guardado como {model_file}")

            results_log.append({
                "datetime"            : datetime.datetime.now().isoformat(sep=" ", timespec="seconds"),
                "batch"               : batch_size,
                "img_size"            : img_size,
                "memoria_gb_vram"     : vram_gb,
                "tiempo_entrenamiento": round(t_train / 60, 1),
                "mAP50"               : round(perf_map50, 4) if perf_map50 else None
            })

    pd.DataFrame(results_log).to_csv(SUMMARY_FILE, index=False)
    print(f"\n📄  Resumen guardado en: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()




