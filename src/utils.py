import numpy as np

def ensure_dirs(cfg):
    (cfg.data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "processed").mkdir(parents=True, exist_ok=True)

def save_data(data, params, cfg):
    ensure_dirs(cfg)
    filename = cfg.data_dir / "processed" / f"r{params[0]}_sigma{params[3]}.npy"
    np.save(filename, data)
