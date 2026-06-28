import os
from datasets import load_from_disk
from huggingface_hub import login

HF_TOKEN = os.environ['HF_TOKEN']
HF_USERNAME = "bdbaburin"

REPO_NAME = "MOMENT-DAPT-Scaling"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

BASE_DIR = r"/home/bdbaburin/datasets/hf_scaling_datasets"


def upload_all():
    login(token=HF_TOKEN)
    
    datasets_to_upload =  ['mixed_0.05B',
                                                         'mixed_0.1B',
                                                         'pure_0.05B',
                                                        'pure_0.1B',]

    for ds_name in datasets_to_upload:
        ds_path = os.path.join(BASE_DIR, ds_name)
        if not os.path.isdir(ds_path): continue
        ds = load_from_disk(ds_path)
        
        try:
            ds.push_to_hub(
                repo_id=REPO_ID,
                config_name=ds_name,
                private=False 
            )
        except Exception as e:
            print(f"Ошибка при загрузке {ds_name}: {e}\n")

if __name__ == "__main__":
    upload_all()
