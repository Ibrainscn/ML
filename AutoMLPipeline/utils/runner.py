import os
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict 
from .custom_dataset import BCIDataset
from .metrics import eval_metric, EarlyStop
from .res_io import ResIO

from torch.utils.tensorboard import SummaryWriter


class Runner:
    def __init__(self, params, data, model):
        super(Runner, self).__init__()
        self.params = params
        self.data = data
        self.model = model
        self.io_writer = ResIO(params)
        self.log_dir = os.path.join(params["mlflow_io"]["output_dir"], "logs")
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        self.all_results = defaultdict(list)
        self.num_epochs = params["mlflow_runner"]["num_epochs"]
        self.batch_size = params["mlflow_runner"]["batch_size"]
        self.num_classes = data["data_shape"]["num_classes"]
       
        self.is_multclass = self.num_classes > 1
        self.predict_model = params["mlflow_runner"]["predict_model"]
        if params["mlflow_runner"]["run_mode"] == "train_test":
            self.do_validation = False
        else:
            self.do_validation = True

        print("Start runner...")
        self.init_device()
        self.init_data_loader()
        self.optimizer = self.init_optimizer()
        self.loss_fxn, self.activation_fxn_name = self.init_loss_activation_fxn()
        # Initialize early stopping with patience=5 and delta=0.001
        self.early_stopping = EarlyStop(patience=5, delta=0.001)


    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
            print("CUDA is available.")
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
        self.model.to(self.device)

    def init_optimizer(self):
        opt_name = self.params["mlflow_runner"]["optimizer"]
        lr = self.params["mlflow_runner"]["lr"]
        weight_decay = self.params["mlflow_runner"].get("weight_decay", 0)
        if opt_name == "sgd":
            opt = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Undefined optimizer name: {opt_name}")

        return opt

    def loss_fxn_argmax(self, pred, targets):
        if len(targets.shape) == 1:
            return self._loss_fxn(pred, targets) 
        elif len(targets.shape) == 2:
            return self._loss_fxn(pred, np.argmax(targets, axis=1))
        else:
            raise ValueError(f"Unexpected targets shape: {targets.shape}")

    def init_loss_activation_fxn(self):
        loss_name = self.params["mlflow_runner"]["loss"]
        loss_params = self.params["mlflow_runner"].get("loss_params", None)
        if loss_params:
            pos_weight = loss_params.get("pos_weight")
            class_weights = loss_params.get("class_weights")
        else:
            class_weights = None
        if loss_name == "mse":
            loss_fxn = nn.MSELoss()
            activation_fxn_name = "none"
        elif loss_name == "mae":
            loss_fxn = nn.L1Loss()
            activation_fxn_name = "none"
        elif loss_name == "bcn":
            loss_fxn = nn.BCEWithLogitsLoss()
            activation_fxn_name = "sigmoid"
        elif loss_name == "bce":
            loss_fxn = nn.BCELoss()
            activation_fxn_name = "sigmoid"
        elif loss_name == "bcn_with_weight":
            loss_fxn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            activation_fxn_name = "sigmoid"
        elif loss_name == "cross_entropy":
            if class_weights:
                self._loss_fxn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
            else:
                self._loss_fxn = nn.CrossEntropyLoss()
            loss_fxn = self.loss_fxn_argmax
            activation_fxn_name = "exp"
        elif loss_name == "nll":
            if class_weights:
                self._loss_fxn = nn.NLLLoss(weight=torch.tensor(class_weights))
            else:
                self._loss_fxn = nn.NLLLoss()
            loss_fxn = self.loss_fxn_argmax
            activation_fxn_name = "exp"
        else:
            raise ValueError(f"Undefined loss function name: {loss_name}")

        return loss_fxn, activation_fxn_name
    
    def init_data_loader(self):
        train_dataset = BCIDataset(self.data["train_dataset"])
        val_dataset = BCIDataset(self.data["val_dataset"])
        test_dataset = BCIDataset(self.data["test_dataset"])
        self.train_loader = DataLoader(train_dataset, self.batch_size, shuffle=False, drop_last=True)
        self.val_loader = DataLoader(val_dataset, self.batch_size, shuffle=False, drop_last=True)
        self.test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False, drop_last=True)

    def train(self):
        print("Start model training...")
        self.model.train()
        for epoch_i in range(self.num_epochs):
            loss_batch = []
            pred = np.array([])
            targets = np.array([])
            # Train the model with batches
            for bch_i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                x = x.to(dtype=torch.float32)
                y = y.to(dtype=torch.float32)
                res = self.model(x)
                if self.activation_fxn_name == "softmax":
                    res = torch.softmax(res, dim=len(res) - 1)
                elif self.activation_fxn_name == "log_softmax":
                    res = torch.log_softmax(res, dim=len(res) - 1)
                elif self.activation_fxn_name == "exp":
                    res = torch.exp(res)
                elif self.activation_fxn_name == "sigmoid":
                    res = torch.sigmoid(res)

                loss = self.loss_fxn(res.squeeze(), y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_batch.append(loss.item())
                pred = np.vstack((pred, res.cpu().detach().numpy())) if pred.size else res.cpu().detach().numpy()
                targets = np.hstack((targets, y.numpy())) if targets.size else y.numpy()

            train_loss = np.mean(loss_batch)
            self.all_results["train_loss"].append(train_loss)
           
            train_metric = eval_metric(self.params, pred.squeeze(), targets)
            self.all_results["train_metric"].append(train_metric[0])

            stdout_str = f"\nTrain    Epoch {epoch_i + 1} / {self.num_epochs}" 
            stdout_str += f"    Loss {train_loss:0.4f}" 
            stdout_str += f"    Metric {train_metric[0]:0.4f}" 
            sys.stdout.write(stdout_str) 
            sys.stdout.flush()

            # Check if training should be stopped
            if self.early_stopping(train_loss):
                print("Early stopping triggered. Stopping training.")
                # save best model and checkpoint for prediction
                self.io_writer.save_model(self.params, self.model, "best", self.device_type)
                self.io_writer.save_checkpoint(self.params, self.model, self.optimizer, "best", self.all_results)

                best_metric = eval_metric(self.params, pred.squeeze(), targets)
                self.all_results["best_metric"].append(best_metric[0])
                self.all_results["best_loss"].append(train_loss)
                self.all_results["best_epoch"].append(epoch_i+1)
                break

            if self.do_validation:
                val_freq = self.params["mlflow_runner"]["val_freq"]
                if (epoch_i + 1) % val_freq == 0:
                    val_loss, val_metric = self.evaluate("val", epoch_i)
                    test_loss, test_metric = self.evaluate("test", epoch_i)

            if (epoch_i + 1) % self.params["mlflow_save"]["save_freq"] == 0:
                self.io_writer.save_model(self.params, self.model, str(epoch_i+1), self.device_type)
                self.io_writer.save_checkpoint(self.params, self.model, self.optimizer, str(epoch_i+1), self.all_results)

            # add loss to tensorboard
            self.tb_writer.add_scalars(
                'Loss', {'train_loss': train_loss,
                         'val_loss': val_loss,
                         'test_loss': test_loss,
                         }, epoch_i)
            # add metric to tensorboard
            self.tb_writer.add_scalars(
                'Metric', {'train_metric': train_metric[0],
                         'val_metric': val_metric[0],
                         'test_metric': test_metric[0],
                         }, epoch_i)
        self.tb_writer.close()
            
            
    def evaluate(self, dataset_key, epoch_i):
        if dataset_key == "val":
            ds_loader = self.val_loader
        elif dataset_key == "test":
            ds_loader = self.test_loader
        else:
            ds_loader = self.train_loader
        
        self.model.eval()
        with torch.no_grad():
            loss_batch = []
            pred = np.array([])
            targets = np.array([])
            for bch_i, (x, y) in enumerate(ds_loader):
                x = x.to(self.device)
                x = x.to(dtype=torch.float32)
                y = y.to(dtype=torch.float32)
                res = self.model(x)
                if self.activation_fxn_name == "softmax":
                    res = torch.softmax(res, dim=len(res) - 1)
                elif self.activation_fxn_name == "log_softmax":
                    res = torch.log_softmax(res, dim=len(res) - 1)
                elif self.activation_fxn_name == "exp":
                    res = torch.exp(res)
                elif self.activation_fxn_name == "sigmoid":
                    res = torch.sigmoid(res)
                
                loss = self.loss_fxn(res.squeeze(), y)

                loss_batch.append(loss.item())
                pred = np.vstack((pred, res.cpu().detach().numpy())) if pred.size else res.cpu().detach().numpy()
                targets = np.hstack((targets, y.numpy())) if targets.size else y.numpy()

            avg_loss = np.mean(loss_batch)
            self.all_results[dataset_key+"_loss"].append(avg_loss)
            evaluate_metric = eval_metric(self.params, pred.squeeze(), targets)
            self.all_results[dataset_key+"_metric"].append(evaluate_metric[0])

            stdout_str = f"\n{dataset_key.capitalize()}   Epoch {epoch_i + 1} / {self.num_epochs}" 
            stdout_str += f"    Loss {avg_loss:0.4f}" 
            stdout_str += f"    Metric {evaluate_metric[0]:0.4f}" 
            sys.stdout.write(stdout_str) 
            sys.stdout.flush()
   
        self.model.train()

        return avg_loss, evaluate_metric


    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for model_id in self.predict_model:
                # load trained model
                wts = self.io_writer.load_model(self.params, model_id, self.device_type)
                self.model.load_state_dict(wts)

                loss_batch = []
                pred = np.array([])
                targets = np.array([])
                for bch_i, (x, y) in enumerate(self.test_loader):
                    x = x.to(self.device)
                    x = x.to(dtype=torch.float32)
                    y = y.to(dtype=torch.float32)
                    res = self.model(x)
                    if self.activation_fxn_name == "softmax":
                        res = torch.softmax(res, dim=len(res) - 1)
                    elif self.activation_fxn_name == "log_softmax":
                        res = torch.log_softmax(res, dim=len(res) - 1)
                    elif self.activation_fxn_name == "exp":
                        res = torch.exp(res)
                    elif self.activation_fxn_name == "sigmoid":
                        res = torch.sigmoid(res)
                    
                    loss = self.loss_fxn(res.squeeze(), y)

                    loss_batch.append(loss.item())
                    pred = np.vstack((pred, res.cpu().detach().numpy())) if pred.size else res.cpu().detach().numpy()
                    targets = np.hstack((targets, y.numpy())) if targets.size else y.numpy()

                avg_loss = np.mean(loss_batch)
                self.all_results["predict_loss"].append(avg_loss)
                evaluate_metric = eval_metric(self.params, pred.squeeze(), targets)
                self.all_results["predict_metric"].append(evaluate_metric[0])
                               
                # save model prediction output
                if self.params["mlflow_save"].get("save_prediction_results", False):
                    dicts = {
                        "y_pred": pred.squeeze(),
                        "y": targets,
                        "all_results": self.all_results,
                    }
                    self.io_writer.save_prediction_results(self.params, dicts, "final_prediction_" + model_id)            