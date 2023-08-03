from datasets import load_dataset
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(dataset_name):
    """Loads the dataset from Hugging Face dataset library and returns the train, test and validation splits"""
    data = load_dataset(dataset_name)
    return data['train'], data['validation'], data['test']


def getDataLoader(frame, tokenizer, batch_size):

    labels = torch.tensor(frame['label'].tolist()).to(torch.int64)

    dataset_sentences = tokenizer(frame['premise'].tolist(), frame['hypothesis'].tolist(), padding=True, truncation=True, return_tensors="pt")
    dataset_tensor = TensorDataset(dataset_sentences.input_ids, dataset_sentences.attention_mask, dataset_sentences.token_type_ids, labels)

    data_loader = DataLoader(dataset_tensor, batch_size)

    return data_loader