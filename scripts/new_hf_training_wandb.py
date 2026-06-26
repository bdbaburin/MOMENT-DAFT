import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import wandb  
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from dotenv import load_dotenv
from momentfm.utils.masking import Masking

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_builder import get_moment_lora


class MOMENTDataset(Dataset):
    def __init__(self, hf_split, seq_len=512, is_train=True):
        self.dataset = hf_split
        self.seq_len = seq_len
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        while True:
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

            input_mask = ~np.isnan(window)
            
            if input_mask.sum() >= (self.seq_len // 2):
                valid_data = window[input_mask]
                if np.std(valid_data) > 1e-5:
                    window = np.nan_to_num(window, nan=0.0)
                    return {
                        "x_enc": torch.tensor(window, dtype=torch.float32).unsqueeze(0),
                        "input_mask": torch.tensor(input_mask, dtype=torch.long)
                    }
            
            idx = random.randint(0, len(self.dataset) - 1)


class MOMENTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from momentfm.utils.masking import Masking
        self.mask_generator = Masking(mask_ratio=0.3, patch_len=8)
        self.cumulative_patches = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        X = inputs.get("x_enc")
        
        input_mask = inputs.get("input_mask")
        if input_mask is None:
            input_mask = inputs.get("real_mask")
            
        if input_mask.dim() == 3:
            input_mask = input_mask.squeeze(1)

        patch_view = input_mask.unfold(dimension=-1, size=8, step=8)
        healthy_patches = (patch_view.sum(dim=-1) == 8).long()
        empty_samples = (healthy_patches.sum(dim=-1) == 0)
        
        if empty_samples.any():
            input_mask[empty_samples, :8] = 1
            X[empty_samples, 0, :8] = 0.0
            
        moment_mask = self.mask_generator.generate_mask(x=X, input_mask=input_mask).to(X.device)
        outputs = model(x_enc=X, input_mask=input_mask, mask=moment_mask)
        
        loss_mask = (input_mask == 1) & (moment_mask == 0)
        
        if loss_mask.sum() > 0:
            raw_mse_no_red = F.mse_loss(outputs.reconstruction.squeeze(1).float(), X.squeeze(1).float(), reduction='none')
            
            X_sq = X.squeeze(1).float()
            mask_fl = input_mask.float()
            mean = (X_sq * mask_fl).sum(dim=1, keepdim=True) / mask_fl.sum(dim=1, keepdim=True).clamp(min=1.0)
            var = (((X_sq - mean)**2) * mask_fl).sum(dim=1, keepdim=True) / mask_fl.sum(dim=1, keepdim=True).clamp(min=1.0)
            var = var.clamp(min=1e-5)
            
            scaled_mse_no_red = raw_mse_no_red / var
            
            loss_raw = (raw_mse_no_red * loss_mask.float()).sum() / loss_mask.sum()
            loss_scaled = (scaled_mse_no_red * loss_mask.float()).sum() / loss_mask.sum()
        else:
            loss_raw = (outputs.reconstruction.sum() * 0.0)
            loss_scaled = torch.tensor(0.0, device=X.device)

        if model.training and wandb.run is not None:
            self.cumulative_patches += X.shape[0] * 64
            
            # Проверяем, что текущий шаг кратен logging_steps
            if self.state.global_step % self.args.logging_steps == 0:
                wandb.log({
                    "moment_metrics/train_loss_scaled": loss_scaled.item(),
                    "moment_metrics/train_loss_raw": loss_raw.item(),
                    "moment_metrics/processed_patches_billions": self.cumulative_patches / 1e9
                }, commit=False)

        return (loss_raw, outputs) if return_outputs else loss_raw


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            X = inputs.get("x_enc")
            input_mask = inputs.get("input_mask")
            if input_mask is None:
                input_mask = inputs.get("real_mask")
            if input_mask.dim() == 3:
                input_mask = input_mask.squeeze(1)

            patch_view = input_mask.unfold(dimension=-1, size=8, step=8)
            healthy_patches = (patch_view.sum(dim=-1) == 8).long()
            empty_samples = (healthy_patches.sum(dim=-1) == 0)
            
            if empty_samples.any():
                input_mask[empty_samples, :8] = 1
                X[empty_samples, 0, :8] = 0.0

            moment_mask = self.mask_generator.generate_mask(x=X, input_mask=input_mask).to(X.device)
            outputs = model(x_enc=X, input_mask=input_mask, mask=moment_mask)
            
            loss_mask = (input_mask == 1) & (moment_mask == 0)
            
            if loss_mask.sum() > 0:
                raw_mse_no_red = F.mse_loss(outputs.reconstruction.squeeze(1).float(), X.squeeze(1).float(), reduction='none')
                
                X_sq = X.squeeze(1).float()
                mask_fl = input_mask.float()
                mean = (X_sq * mask_fl).sum(dim=1, keepdim=True) / mask_fl.sum(dim=1, keepdim=True).clamp(min=1.0)
                var = (((X_sq - mean)**2) * mask_fl).sum(dim=1, keepdim=True) / mask_fl.sum(dim=1, keepdim=True).clamp(min=1.0)
                var = var.clamp(min=1e-5)
                
                scaled_mse = raw_mse_no_red / var
                eval_loss = (scaled_mse * loss_mask.float()).sum() / loss_mask.sum()
            else:
                eval_loss = torch.tensor(0.0).to(X.device)
                
        return (eval_loss, None, None)


def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    os.environ["WANDB_PROJECT"] = "moment-lora-tuning"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint" 
    
    print("[*] Загрузка датасета...")
    hf_dataset = load_dataset(
        "bdbaburin/MOMENT-DAPT-Scaling", 
        name="mixed_0.5B", 
        num_proc=8,
        token=hf_token
    )
    
    train_dataset = MOMENTDataset(hf_dataset["train"], is_train=True)
    eval_dataset = MOMENTDataset(hf_dataset["validation"], is_train=False)
    
    print("[*] Инициализация модели...")
    model = get_moment_lora(model_name="AutonLab/MOMENT-1-small")
    model.print_trainable_parameters()
    
    if hasattr(model, "config") and not hasattr(model.config, "to_dict"):
        model.config.__class__.to_dict = lambda self: vars(self)
    
    training_args = TrainingArguments(
        output_dir="./local_sanity_check_checkpoints",
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        learning_rate=1e-5,              
        warmup_steps=20,
        max_grad_norm=5.0,               
        num_train_epochs=2,
        bf16=True,                       
        logging_steps=10,             
        

        eval_strategy="steps",           
        eval_steps=50,                 
        save_strategy="steps",           
        save_steps=50,                  
        load_best_model_at_end=True,     
        metric_for_best_model="eval_loss", 
        greater_is_better=False,                
        
        dataloader_num_workers=6,
        remove_unused_columns=False,     
        report_to="wandb",             
        save_total_limit=2               
    )

    trainer = MOMENTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("[*] Запуск локального Trainer...")
    trainer.train()

if __name__ == "__main__":
    main()
    