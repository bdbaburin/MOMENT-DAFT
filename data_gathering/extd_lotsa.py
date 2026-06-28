import os
import shutil
from datasets import load_dataset


BASE_DIR = r"./lotsa_project_data"   


CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
DOMAIN_SAVE_DIR = os.path.join(BASE_DIR, "business_domain")
OOD_SAVE_DIR = os.path.join(BASE_DIR, "ood_replay_buffer")

EXTRA_BUSINESS = [
    "buildings_900k",                     
    "extended_web_traffic_with_missing",  
    "traffic_hourly", "traffic_weekly",   
    "monash_m3_monthly", "monash_m3_quarterly", "monash_m3_yearly", "monash_m3_other",
    "nn5_daily_with_missing", "nn5_weekly",
    "pedestrian_counts"                  
]

EXTRA_OOD = [
    "temperature_rain_with_missing",      
    "wind_farms_with_missing",           
    "era5_2018",                          
    "covid_deaths", "covid_mobility"      
]

def download_and_clean(config_name, save_dir):
    save_path = os.path.join(save_dir, config_name)
    
    if os.path.exists(save_path):
        return

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
    except Exception as e:
        print(f" Ошибка при загрузке {config_name}: {e}")

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    for config in EXTRA_BUSINESS:
        download_and_clean(config, DOMAIN_SAVE_DIR)

    for config in EXTRA_OOD:
        download_and_clean(config, OOD_SAVE_DIR)
        
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        

if __name__ == "__main__":
    main()