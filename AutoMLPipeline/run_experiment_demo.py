import os
import json
import shutil
import numpy as np
import pandas as pd
from tensorboard import program
from utils.custom_dataset import data_preprocess
from utils.nn_lstm import LSTM
from utils.nn_tcn import TCN
from utils.runner import Runner


def setup_config(config_file):
    with open(config_file) as f:
        params = json.load(f)

    return params


def setup_dataset(data, params):

    return data_preprocess(data, params)


def setup_model(params, data_shape):
    model_name = params["mlflow_model"]["model_name"]
    model_params = params["mlflow_model"]["model_params"]
    win_size = params["mlflow_data"]["win_size"]
    num_features = data_shape["num_features"]
    num_classes = data_shape["num_classes"]
    # input shape: batch_size, win_size, num_features
    input_shape = (1, win_size, num_features)
    if model_name == "LSTM":
        model = LSTM(input_shape, model_params, num_classes)
    elif model_name == "TCN":
        model = TCN(input_shape, model_params, num_classes)
    else:
        raise ValueError("Model {model_name} has not implemented yet!")
    
    return model


def setup_expriment(params, data, model):

    return Runner(params, data, model)


def launch_tensorboard(logdir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', '6006'])
    url = tb.launch()
    print(f"TensorBoard started at {url}")


if __name__ == "__main__":
    print("Start running the experiment...")

    # Setup config
    config_file = "config/example_config.json"
    params = setup_config(config_file)

    # Setup dataset
    # Example 2D array
    arr_Xy = np.random.randint(0, 2, size=(1000, 4))
    data = setup_dataset(arr_Xy, params)

    # Setup model
    model = setup_model(params, data["data_shape"])

    # Setup experiment
    exp = setup_expriment(params, data, model)

    # Remove any previous old logs
    log_dir = os.path.join(params["mlflow_io"]["output_dir"], "logs")
    shutil.rmtree(log_dir)

    # Run experiment
    # Train and validate the model
    exp.train()

    # Predict
    exp.predict()

    # Launch TensorBoard to visualize the train/val/test loss and metric
    launch_tensorboard(log_dir)
    # Or run the following command in your terminal, then follow the link
    # tensorboard --logdir=exp_output_demo/logs --port 8000