import torch
import numpy as np
import random
from torch.utils.data import Dataset

class MOMENTDataset(Dataset):
    def __init__(self, hf_split, seq_len=512, is_train=True):
        self.dataset = hf_split
        self.seq_len = seq_len
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Архитектурный фильтр: перебираем, пока не найдем ряд по стандартам MOMENT
        for _ in range(100):
            ts = np.array(self.dataset[idx]["target"], dtype=np.float32)
            if ts.ndim > 1: ts = ts.flatten()
            ts_len = len(ts)

            if ts_len > self.seq_len:
                start_idx = random.randint(0, ts_len - self.seq_len) if self.is_train else ts_len - self.seq_len
                window = ts[start_idx : start_idx + self.seq_len]
            elif ts_len < self.seq_len:
                pad_len = self.seq_len - ts_len
                window = np.pad(ts, (pad_len, 0), constant_values=np.nan)
            else:
                window = ts.copy()

            # 1 - данные есть, 0 - NaN/паддинг
            input_mask = ~np.isnan(window)
            
            # --- СТРОГО ПО СТАТЬЕ MOMENT (Раздел 3.1): ---
            # 1. discard sequences with fewer than L/2 observations
            if input_mask.sum() >= (self.seq_len // 2):
                valid_data = window[input_mask]
                
                # 2. or with constant values (Защита RevIN от деления на 0)
                if np.std(valid_data) > 1e-5:
                    window = np.nan_to_num(window, nan=0.0)
                    return {
                        "x_enc": torch.tensor(window, dtype=torch.float32).unsqueeze(0), # [1, 512]
                        "input_mask": torch.tensor(input_mask, dtype=torch.long)         # [512]
                    }
            
            # Если ряд не прошел — берем другой
            idx = random.randint(0, len(self.dataset) - 1)
            
        return self.__getitem__(random.randint(0, len(self.dataset) - 1))
