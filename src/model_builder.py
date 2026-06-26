import torch
from transformers import AutoModel
from momentfm import MOMENTPipeline
from peft import LoraConfig, get_peft_model

def get_moment_lora(model_name="AutonLab/MOMENT-1-small", r=16, alpha=32, dropout=0.05):
    pipeline = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "reconstruction", #
        }
    )
    pipeline.init() 
    
    lora_config = LoraConfig(
        r=r, 
        lora_alpha=alpha,
        # target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"], 
        target_modules=r".*\.(q|k|v|o|wi_0|wi_1|wo)$",
        lora_dropout=dropout,
        bias="none", 
        task_type="FEATURE_EXTRACTION",
        modules_to_save=["reconstruction_head", "head"] 
    )
    model = get_peft_model(pipeline, lora_config)

    for name, param in model.named_parameters():
        if "reconstruction_head" in name or "head" in name:
            param.data = param.data.float() 
            
    return model
