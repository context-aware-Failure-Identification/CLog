from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np

TORCH_INT_TYPE = torch.int16
NP_INT_TYPE = np.int16


def create_train_valid_data_loaders(load_train, labels_train, load_test, labels_test, pad_len, batch_size):
    train_data = TensorDataset(torch.tensor(get_padded_data(load_train, pad_len=pad_len), dtype=torch.long), torch.tensor(labels_train.astype(np.long), dtype=torch.long))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


    test_data = TensorDataset(torch.tensor(get_padded_data(load_test, pad_len=pad_len), dtype=torch.long), torch.tensor(labels_test.astype(np.long).flatten(), dtype=torch.long))
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return train_dataloader, test_dataloader


def create_test_data_loader(load_test, labels_test, pad_len, batch_size):
    test_data = TensorDataset(torch.tensor(get_padded_data(load_test, pad_len=pad_len), dtype=torch.long), torch.tensor(labels_test.astype(np.long).flatten(), dtype=torch.long))
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return test_dataloader


def get_padded_data(data, pad_len):
    pd = pad_sequences(data, maxlen=pad_len, dtype="long", truncating="post", padding="post")
    return pd

