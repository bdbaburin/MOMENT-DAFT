import os
import shutil
from datasets import load_dataset


BASE_DIR = "~/datasets" 
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache") 
DOMAIN_SAVE_DIR = os.path.join(BASE_DIR, "business_domain")
OOD_SAVE_DIR = os.path.join(BASE_DIR, "ood_replay_buffer")

DOMAIN_WHITELIST = [
    "m1_monthly", "m1_quarterly", "m1_yearly",
    "m4_hourly", "m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly", "m4_yearly",
    "m5", "favorita_sales", "favorita_transactions", 
    "hierarchical_sales", "restaurant",
    "tourism_monthly", "tourism_quarterly", "tourism_yearly",
    "alibaba_cluster_trace_2018", "azure_vqm_traces_2017", "borg_cluster_data_2011",
    "kaggle_web_traffic_weekly", "wiki-rolling_nips",
    "london_smart_meters_with_missing", "residential_load_power"
]

OOD_WHITELIST = [
    "weather",               
    "cdc_fluview_ilinet",    
    "hospital",              
    "sunspot_with_missing", 
    "bitcoin_with_missing"  
]

def download_and_clean(config_name, save_dir):
    save_path = os.path.join(save_dir, config_name)
    
    if os.path.exists(save_path):
        print(f"  [ПРОПУСК] {config_name} уже лежит в {save_path}")
        return

    print(f"  [*] Загрузка: {config_name} ...")
    try:
        dataset = load_dataset(
            "Salesforce/lotsa_data", 
            name=config_name, 
            split="train", 
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        
        dataset.save_to_disk(save_path)
        dataset.cleanup_cache_files()
        
        print(f" Успех! Сохранено временных рядов: {len(dataset)}")
        
    except Exception as e:
        print(f" Ошибка при загрузке {config_name}: {e}")

def main():
    # Создаем нужные папки на большом диске
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DOMAIN_SAVE_DIR, exist_ok=True)
    os.makedirs(OOD_SAVE_DIR, exist_ok=True)
    
    for config in DOMAIN_WHITELIST:
        download_and_clean(config, DOMAIN_SAVE_DIR)
        
    for config in OOD_WHITELIST:
        download_and_clean(config, OOD_SAVE_DIR)

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()