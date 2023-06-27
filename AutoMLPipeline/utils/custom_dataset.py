import time
import numpy as np
from torch.utils.data import Dataset


def sliding_window(arr, win_size, stride=1):
    # separte X, y
    X, y = arr[:, 1:], arr[:, 0]
    feature, label = [], []
    for i in range(0, X.shape[0] - win_size + 1, stride):
        feature.append(X[i:i+win_size])
        # use the last value of the window as label
        label.append(y[i+win_size-1])

    return np.array(feature), np.array(label)


def train_val_test_split(arr, split_ratio):
    # Split the time series data
    num_len = arr.shape[0]
    train_data = arr[int(split_ratio[0][0] * num_len):int(split_ratio[0][1] * num_len)]
    val_data = arr[int(split_ratio[1][0] * num_len):int(split_ratio[1][1] * num_len)]
    test_data = arr[int(split_ratio[2][0] * num_len):int(split_ratio[2][1] * num_len)]
    # Print the sizes of the train, validation, and test sets
    print(f"Train set size: {train_data.shape}")
    print(f"Validation set size: {val_data.shape}")
    print(f"Test set size: {test_data.shape}")

    return train_data, val_data, test_data


def data_preprocess(arr, params):
    data_params = params["mlflow_data"]
    train_data, val_data, test_data = train_val_test_split(arr, data_params["train_val_test_split_ratio"])
    data = {}
    # setup train dataset
    X, y = sliding_window(train_data, data_params["win_size"], data_params["stride"])
    data["train_dataset"] = {"X": X, "y": y}
    # setup val dataset
    X, y = sliding_window(val_data, data_params["win_size"], data_params["stride"])
    data["val_dataset"] = {"X": X, "y": y}
    # setup test dataset
    X, y = sliding_window(test_data, data_params["win_size"], data_params["stride"])
    data["test_dataset"] = {"X": X, "y": y}
    # get data shape
    num_classes = y.shape[-1] if y.ndim > 1 else 1
    data_shape = {
        "num_features": X.shape[-1], 
        "num_classes": num_classes, 
        "feature_shape": X[0].shape
    }
    data["data_shape"] = data_shape

    return data


class BCIDataset(Dataset):
    def __init__(self, data):
        super(BCIDataset, self).__init__()
        self.X = data["X"]
        self.y = data["y"]
    
    def __getitem__(self, idx):
        # x = torch.tensor(self.X[idx])
        # y = torch.tensor(self.y[idx])
        x = self.X[idx]
        y = self.y[idx]

        return x, y
    
    def __len__(self):
        return len(self.y)