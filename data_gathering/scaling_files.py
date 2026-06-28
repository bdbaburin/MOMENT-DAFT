import os
import numpy as np
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Features, Sequence, Value


BASE_DIR = r"/home/bdbaburin/datasets" 

DOMAIN_SAVE_DIR = os.path.join(BASE_DIR, "business_domain")
OOD_SAVE_DIR = os.path.join(BASE_DIR, "ood_replay_buffer")

FINAL_OUT_DIR = os.path.join(BASE_DIR, "hf_scaling_datasets")
os.makedirs(FINAL_OUT_DIR, exist_ok=True)

STANDARD_FEATURES = Features({
    "dataset_origin": Value("string"),
    "item_id": Value("string"),
    "ta                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           rget": Sequence(Value("float32"))
})

SCALING_FRACTIONS = {
    "0.1B": 0.05, 
    "0.05B": 0.025,  
    # аналогично указать другие скейлы 
}

def standardize_dataset(ds_path, dataset_name):
    ds = load_from_disk(ds_path)
    
    if dataset_name == "buildings_900k":
        ds = ds.shuffle(seed=42).select(range(min(50000, len(ds))))
        
    def format_batch(batch, indices):
        new_batch = {"dataset_origin": [dataset_name] * len(indices), "item_id": [], "target": []}
        item_ids = batch.get("item_id", [f"{dataset_name}_entity_{i}" for i in indices])
        
        for i, target_array in enumerate(batch["target"]):
            new_batch["item_id"].append(str(item_ids[i]))
            arr = np.array(target_array, dtype=np.float32)
            if arr.ndim > 1: arr = arr.flatten()
            new_batch["target"].append(arr.tolist())
            
        return new_batch

    return ds.map(
        format_batch, with_indices=True, batched=True, batch_size=1000,
        remove_columns=ds.column_names, features=STANDARD_FEATURES, desc=f"Format {dataset_name}"
    )

def main():
    pure_train_splits = {size: [] for size in SCALING_FRACTIONS}
    ood_train_splits = {size: [] for size in SCALING_FRACTIONS}
    business_val = [] 
    for ds_name in sorted(os.listdir(DOMAIN_SAVE_DIR)):
        ds_path = os.path.join(DOMAIN_SAVE_DIR, ds_name)
        if not os.path.isdir(ds_path): continue
            
        std_ds = standardize_dataset(ds_path, ds_name)
        
        if len(std_ds) < 5:
            for size in SCALING_FRACTIONS:
                pure_train_splits[size].append(std_ds)
            continue

        split = std_ds.train_test_split(test_size=0.20, seed=42)
        train_full = split["train"]

        business_val.append(split["test"])
        
        for size, frac in SCALING_FRACTIONS.items():
            num_rows = max(1, int(len(train_full) * frac))
            subset_train = train_full.shuffle(seed=42).select(range(num_rows))
            pure_train_splits[size].append(subset_train)
            
    for ds_name in sorted(os.listdir(OOD_SAVE_DIR)):
        ds_path = os.path.join(OOD_SAVE_DIR, ds_name)
        if not os.path.isdir(ds_path): continue
            
        std_ds = standardize_dataset(ds_path, ds_name)

        for size, frac in SCALING_FRACTIONS.items():
            num_rows = max(1, int(len(std_ds) * frac))
            subset_ood = std_ds.shuffle(seed=42).select(range(num_rows))
            ood_train_splits[size].append(subset_ood)
            
    shared_val_ds = concatenate_datasets(business_val)

    for size in SCALING_FRACTIONS:
        pure_train_ds = concatenate_datasets(pure_train_splits[size]).shuffle(seed=42)
        ds_pure = DatasetDict({"train": pure_train_ds, "validation": shared_val_ds})
        
        pure_path = os.path.join(FINAL_OUT_DIR, f"pure_{size}")
        ds_pure.save_to_disk(pure_path)

        ood_train_ds = concatenate_datasets(ood_train_splits[size])
        mixed_train_ds = concatenate_datasets([pure_train_ds, ood_train_ds]).shuffle(seed=42)
        ds_mixed = DatasetDict({"train": mixed_train_ds, "validation": shared_val_ds})
        
        mixed_path = os.path.join(FINAL_OUT_DIR, f"mixed_{size}")
        ds_mixed.save_to_disk(mixed_path)


if __name__ == "__main__":
    main()