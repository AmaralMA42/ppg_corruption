import json
from dataclasses import asdict
from datetime import datetime

import numpy as np

def make_run_id():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def config_metadata(cfg, experiment, **extra):
    metadata = asdict(cfg)
    metadata["experiment"] = experiment
    metadata["passos_media"] = cfg.passos_media
    metadata.update(extra)
    return metadata

def save_npz_result(cfg, category, name, metadata=None, **arrays):
    output_dir = cfg.data_dir / "processed" / category
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = make_run_id()
    filename = output_dir / f"{name}_{run_id}.npz"
    payload = dict(arrays)
    metadata = dict(metadata or {})
    metadata["created_at"] = run_id
    metadata["category"] = category
    metadata["filename"] = filename.name
    payload["metadata"] = json.dumps(metadata, ensure_ascii=True)

    np.savez_compressed(filename, **payload)
    return filename

def load_npz_result(path):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(data["metadata"].item()) if "metadata" in data.files else {}
    arrays = {key: data[key] for key in data.files if key != "metadata"}
    return arrays, metadata
