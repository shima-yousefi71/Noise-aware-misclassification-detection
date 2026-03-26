import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def normalize_data(data, file_name=None):
    if data.ndim < 3:
        return np.zeros_like(data), np.array(0.0), np.array(1.0)

    min_val = np.min(data, axis=(0, 1, 2), keepdims=True)
    max_val = np.max(data, axis=(0, 1, 2), keepdims=True)
    range_val = max_val - min_val

    epsilon = 1e-6
    range_val = np.maximum(range_val, epsilon)

    normalized_data = (data - min_val) / range_val
    return normalized_data, min_val, max_val


def denormalize_data(normalized_data, min_val, max_val):
    min_val = min_val.cpu().numpy() if isinstance(min_val, torch.Tensor) else min_val
    max_val = max_val.cpu().numpy() if isinstance(max_val, torch.Tensor) else max_val
    return normalized_data * (max_val - min_val) + min_val


class ToTensor:
    def __call__(self, sample):
        return sample.clone().detach().float()


class UnlabeledDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.min_max_vals = []
        self.file_names = []
        self.problematic_files = []

        for file in tqdm(os.listdir(data_dir), desc="Loading Data"):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "rb") as f:
                        content = pickle.load(f)

                    if isinstance(content, dict) and "data" in content:
                        data = content["data"]
                    elif isinstance(content, np.ndarray):
                        data = content
                    else:
                        raise ValueError(f"Unexpected format in file: {file_path}")

                    normalized_data, min_val, max_val = normalize_data(data, file_name=file_path)
                    self.data.append(normalized_data)
                    self.min_max_vals.append((min_val, max_val))
                    self.file_names.append(file)

                except Exception:
                    self.problematic_files.append(file_path)

        if self.data:
            self.data = torch.tensor(np.stack(self.data), dtype=torch.float32)
        else:
            raise ValueError("No valid data files were loaded. Check the log for problematic files.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        min_val, max_val = self.min_max_vals[idx]
        file_name = self.file_names[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, min_val, max_val, file_name

    def save_problematic_files(self, log_path="problematic_files.log"):
        with open(log_path, "w") as log_file:
            for file in self.problematic_files:
                log_file.write(f"{file}\n")
        print(f"Problematic files saved to {log_path}")
