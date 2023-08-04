import pandas as pd
from datasets import load_dataset
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import configparser


def load_data(dataset_name):
    """Loads the dataset from Hugging Face hub using dataset library and returns the train, test and validation dataFrames"""

    config = configparser.ConfigParser()
    config.read('config.ini')

    data = load_dataset(dataset_name)

    if dataset_name == 'imdb':      # For IMDB dataset, I'm using a custom split instead of the default 25k for test and train

        combined_IMDB_frame = pd.concat([data['train'].to_pandas(), data['test'].to_pandas()])  # Combines the train and test splits
        combined_IMDB_frame = combined_IMDB_frame.sample(frac=1, random_state=45)

        train_ratio = config.getfloat('IMDB', 'train_ratio')    # Loading the train, test and val split ratios from config file
        test_ratio = config.getfloat('IMDB', 'test_ratio')
        val_ratio = config.getfloat('IMDB', 'val_ratio')

        # Creating the train, test and validation data frames

        train_data, temp_data = train_test_split(combined_IMDB_frame, test_size=1 - train_ratio, random_state=45)
        val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (test_ratio + val_ratio), random_state=45)

        return train_data, val_data, test_data

    elif dataset_name == 'rotten_tomatoes':     # For rotten_tomatoes dataset, I'm using the default split

        return data['train'].to_pandas(), data['validation'].to_pandas(), data['test'].to_pandas()


def getDataLoader(frame, tokenizer, batch_size):

    labels = torch.tensor(frame['label'].tolist()).to(torch.int64)

    dataset_sentences = tokenizer(frame['premise'].tolist(), frame['hypothesis'].tolist(), padding=True, truncation=True, return_tensors="pt")
    dataset_tensor = TensorDataset(dataset_sentences.input_ids, dataset_sentences.attention_mask, dataset_sentences.token_type_ids, labels)

    data_loader = DataLoader(dataset_tensor, batch_size)

    return data_loader
