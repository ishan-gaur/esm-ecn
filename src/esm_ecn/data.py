import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from datasets import load_dataset
from esm_ecn.constants import DATA_FOLDER

def preprocess_embeddings_and_labels(data_split, model_name="esmc_300m", n_shard=10000):
    esm = ESMC.from_pretrained(model_name).to("cuda")
    ds = load_dataset("lightonai/SwissProt-EC-leaf")
    ds = ds.with_format("torch")

    all_splits = ["train", "dev", "test"]
    if data_split not in all_splits:
        raise ValueError(f"data_split must be one of {all_splits}")

    X_data, y_data = ds[data_split]['seq'], ds[data_split]['labels']

    y_max = max([y for split in all_splits for y_list in ds[split]['labels'] for y in y_list]) # dev and test are missing one label
    y_data = [torch.nn.functional.one_hot(y, y_max+1).sum(0) for y in y_data]
    y_data = torch.stack(y_data)
    torch.save(y_data, DATA_FOLDER / f"{data_split}_labels.pt")

    logits_config = LogitsConfig(sequence=True, return_embeddings=True)    

    existing_shards = list(DATA_FOLDER.glob(f"{model_name}_{data_split}_cls_embeddings_*.pt"))
    existing_shard_numbers = {int(str(f).split("_")[-1].split(".")[0]) for f in existing_shards}
    total_shards = (len(X_data) + n_shard - 1) // n_shard
    missing_shards = set(range(total_shards)) - existing_shard_numbers

    if not missing_shards:
        print("All shards are already processed.")
        return

    first_missing_shard = min(missing_shards)
    X_data = X_data[first_missing_shard * n_shard:]

    cls_embeddings = []
    res_avg_embeddings = []
    shard_num = first_missing_shard
    for X_batch in tqdm(X_data, desc="Embedding"):
        encoded_X = esm.encode(ESMProtein(sequence=X_batch))
        logits = esm.logits(encoded_X, logits_config)
        cls_embeddings.append(logits.embeddings[:, 0]) # embedding dim 960
        res_avg_embeddings.append(logits.embeddings[:, 1:-1].mean(1))

        if len(cls_embeddings) >= n_shard:
            cls_embeddings = torch.cat(cls_embeddings)
            res_avg_embeddings = torch.cat(res_avg_embeddings)
            torch.save(cls_embeddings, DATA_FOLDER / f"{model_name}_{data_split}_cls_embeddings_{shard_num}.pt")
            torch.save(res_avg_embeddings, DATA_FOLDER / f"{model_name}_{data_split}_res_avg_embeddings_{shard_num}.pt")
            shard_num += 1
            cls_embeddings = []
            res_avg_embeddings = []

    if len(cls_embeddings) > 0:
        cls_embeddings = torch.cat(cls_embeddings)
        res_avg_embeddings = torch.cat(res_avg_embeddings)
        torch.save(cls_embeddings, DATA_FOLDER / f"{model_name}_{data_split}_cls_embeddings_{shard_num}.pt")
        torch.save(res_avg_embeddings, DATA_FOLDER / f"{model_name}_{data_split}_res_avg_embeddings_{shard_num}.pt")


def load_cls_embeddings(model, data_split):
    folder_files = list(DATA_FOLDER.glob(f"{model}_{data_split}_cls_embeddings_*.pt"))
    n_shard = len(folder_files)
    shard_numbers = [int(str(f).split("_")[-1].split(".")[0]) for f in folder_files]
    assert max(shard_numbers) == n_shard - 1
    x = torch.cat([torch.load(DATA_FOLDER / f"{model}_{data_split}_cls_embeddings_{i}.pt", weights_only=False) for i in range(n_shard)])
    print(f"Loaded embeddings, shape: {x.shape}")
    return x

def load_res_avg_embeddings(model, data_split):
    folder_files = list(DATA_FOLDER.glob(f"{model}_{data_split}_res_avg_embeddings_*.pt"))
    n_shard = len(folder_files)
    shard_numbers = [int(str(f).split("_")[-1].split(".")[0]) for f in folder_files]
    assert max(shard_numbers) == n_shard - 1
    x = torch.cat([torch.load(DATA_FOLDER / f"{model}_{data_split}_res_avg_embeddings_{i}.pt", weights_only=False) for i in range(n_shard)])
    print(f"Loaded embeddings, shape: {x.shape}")
    return x

def load_labels(data_split):
    y = torch.load(DATA_FOLDER / f"{data_split}_labels.pt", weights_only=False)
    print(f"Loaded labels, shape: {y.shape}")
    return y

def data_loader(model, batch_size, data_split, shuffle, cls):
    print(f"Loading {data_split} embeddings")
    X = load_cls_embeddings(model, data_split) if cls else load_res_avg_embeddings(model, data_split)
    print(f"Loading {data_split} labels")
    y = load_labels(data_split)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_data_loader(model, batch_size, cls):
    return data_loader(model, batch_size, "train", shuffle=True, cls=cls)

def val_data_loader(model, batch_size, cls):
    return data_loader(model, batch_size, "dev", shuffle=False, cls=cls)

def test_data_loader(model, batch_size, cls):
    return data_loader(model, batch_size, "test", shuffle=False, cls=cls)