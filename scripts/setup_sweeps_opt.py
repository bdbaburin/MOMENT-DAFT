import os
import torch
import gc
import torch.nn.functional as F
import numpy as np
import random
import wandb
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback 
from dotenv import load_dotenv
from momentfm.utils.masking import Masking

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_builder import get_moment_lora

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["WANDB_HTTP_TIMEOUT"] = "60"

os.environ["WANDB_START_METHOD"] = "thread" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

MODELS = ["AutonLab/MOMENT-1-small", "AutonLab/MOMENT-1-base", "AutonLab/MOMENT-1-large"]
DATASETS = ["pure_0.05B", "pure_0.1B", "mixed_0.05B", "mixed_0.1B", "pure_0.25B",  "mixed_0.25B", "pure_0.5B", "mixed_0.5B", 
            "pure_1.0B",  "mixed_1.0B", "pure_2.0B", "mixed_2.0B",]

def train_sweep():
    with wandb.init() as run:
        config = wandb.config 
        
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        
        print(f"[*] Загрузка датасета: {config.dataset_name}")
        full_train = load_dataset(
            "bdbaburin/MOMENT-DAPT-Scaling", 
            name=config.dataset_name, 
            split="train",
            num_proc=8,
            token=hf_token
        )
        full_eval = load_dataset(
            "bdbaburin/MOMENT-DAPT-Scaling", 
            name=config.dataset_name, 
            split="validation",
            num_proc=8,
            token=hf_token
        )
        
        train_15_pct = int(0.35 * len(full_train))
        eval_15_pct = int(0.15 * len(full_eval))

        hf_dataset_train = full_train.shuffle(seed=42).select(range(train_15_pct)).with_format("numpy")
        hf_dataset_eval = full_eval.shuffle(seed=42).select(range(eval_15_pct)).with_format("numpy")
        
        train_dataset = MOMENTDataset(hf_dataset_train, is_train=True)
        eval_dataset = MOMENTDataset(hf_dataset_eval, is_train=False)

        
        current_alpha = 2 * config.lora_r
        model = get_moment_lora(
            model_name=config.model_name, 
            r=config.lora_r, 
            alpha=current_alpha
        )
        
        if hasattr(model, "config") and not hasattr(model.config, "to_dict"):
            model.config.__class__.to_dict = lambda self: vars(self)
        
        training_args = TrainingArguments(
            output_dir="./local_sweep_checkpoints",
            per_device_train_batch_size=1024,
            per_device_eval_batch_size=1024,
            
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            
            warmup_steps=20,
            max_grad_norm=5.0,               
            num_train_epochs=5,
            bf16=True,                       
            logging_steps=10,
            
            eval_strategy="steps",           
            eval_steps=10,                  
            save_strategy="steps",           
            save_steps=10,                  
            load_best_model_at_end=True,     
            metric_for_best_model="eval_loss", 
            greater_is_better=False,         
            
            dataloader_num_workers=6,
            dataloader_persistent_workers=False,
            remove_unused_columns=False,     
            report_to="wandb",
            save_total_limit=1               
        )

        trainer = MOMENTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
        )

        trainer.train()
        
        del trainer
        del model
        del train_dataset
        del eval_dataset
        torch.cuda.empty_cache()
        gc.collect()

def main():
    load_dotenv()
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
        
    os.environ["WANDB_PROJECT"] = "moment-lora-DAPT-v2"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint" 

    sweep_ids_env = os.getenv("SWEEP_IDS")

    if sweep_ids_env:
        sweep_ids = [s.strip() for s in sweep_ids_env.split(",")]
        print(f"[*]Получено очередей для выполнения: {len(sweep_ids)}")
        
        for s_id in sweep_ids:
            print(f"[*] Подключение к очереди: {s_id}")
            wandb.agent(s_id, train_sweep, count=7, project="moment-lora-DAPT-v2")
        
        return

    generated_sweeps = []
    
    for model_name in MODELS:
        for ds_name in DATASETS:
            sweep_config = {
                'method': 'bayes',
                'name': f"sweep_{model_name.split('-')[-1]}_{ds_name}",
                'metric': {'name': 'eval/loss', 'goal': 'minimize'},
                'early_terminate': {
                    'type': 'hyperband',
                    'min_iter': 3,  
                    'eta': 2     
                },
                'parameters': {
                    'model_name': {'value': model_name},
                    'dataset_name': {'value': ds_name},
                    
                    'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 5e-4},
                    'weight_decay': {'distribution': 'uniform', 'min': 0.0, 'max': 0.1},
                    'lora_r': {'values': [8, 16, 32,]},
                    'lr_scheduler_type': {'values': ['linear', 'cosine', 'constant']}
                }
            }
            s_id = wandb.sweep(sweep_config, project="moment-lora-DAPT-v2")
            generated_sweeps.append(s_id)
            print(f"[{model_name.split('-')[-1]} | {ds_name}] -> SWEEP_ID: {s_id}")


    print(",".join(generated_sweeps))


if __name__ == "__main__":
    main()
