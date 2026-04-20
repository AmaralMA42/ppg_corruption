import os
import numpy as np

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def save_data(data, params):
    ensure_dirs()
    filename = f"data/processed/r{params[0]}_sigma{params[3]}.npy"
    np.save(filename, data)