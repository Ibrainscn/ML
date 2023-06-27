import os
import torch
import numpy as np
import glob


class ResIO:
    def __init__(self, params):
        super(ResIO, self).__init__()
        self.params = params


    def save_checkpoint(self, params, model, optimizer, epoch_str, all_results):
        if not params["mlflow_save"].get("save_checkpoint", False):
            return
        # save model checkpoint
        output_dir = os.path.join(params["mlflow_io"]["output_dir"], "checkpoint")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_fname = os.path.join(output_dir, "checkpoint_" + epoch_str + ".pt")
        with open(model_fname, "wb") as f:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch_i": epoch_str,
                    "all_results": all_results,
                    "checkpoint_dir": model_fname
                }, 
                f,
                )
            print(f"Successfully saved checkpoint: {model_fname}")


    def load_checkpoint(self, params, checkpoint_id, device_type="cpu"):
        output_dir = os.path.join(params["mlflow_io"]["output_dir"], "checkpoint")
        device = torch.device(device_type)
        checkpoint_fname = os.path.join(output_dir, "model_" + checkpoint_id + ".pt")
        if not os.path.exists(checkpoint_fname):
            raise ValueError(f"Model does not exist: {checkpoint_fname}")
        else:
            with open(checkpoint_fname, "rb") as f:
                print(f"Loading checkpoint: {checkpoint_fname}")
                model = torch.load(f, map_location=device)

        return model


    def save_model(self, params, model, epoch_str, device_type="cpu"):
        if not params["mlflow_save"].get("save_model", False):
            return
        # save model .pt file
        output_dir = os.path.join(params["mlflow_io"]["output_dir"], "model")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_fname = os.path.join(output_dir, "model_" + epoch_str + ".pt")
        with open(model_fname, "wb") as f:
            torch.save(model.state_dict(), f)
        print(f"\nSuccessfully saved model: {model_fname}")

        # save model with torchscript
        if params["mlflow_save"].get("save_model_torchscript", False):
            device = torch.device(device_type)
            model.eval()
            model.to(device)
            example_inputs = torch.rand(model.data_shape, device=device)
            model(example_inputs)
            torchscript = torch.jit.trace(model, example_inputs)
            model_fname = os.path.join(output_dir, "model_" + epoch_str + "_torchscript_" + device_type + ".pt")
            with open(model_fname, "wb") as f:
                torch.jit.save(torchscript, f)
            print(f"Successfully saved torchscript: {model_fname}")


    def load_model(self, params, model_id, device_type="cpu"):
        output_dir = os.path.join(params["mlflow_io"]["output_dir"], "model")
        device = torch.device(device_type)
        model_fname = os.path.join(output_dir, "model_" + model_id + ".pt")
        if not os.path.exists(model_fname):
            if model_id == "best":
                model_paths = np.sort(glob.glob(f"{output_dir}\\model_*.pt"))
                # get the latest model.pt 
                model_ids = []
                for i in range(len(model_paths)):
                    file_name = os.path.basename(model_paths[i])
                    if "torchscript" not in file_name:
                        model_id = int(file_name.split('_')[1].split('.')[0])
                        # print(file_name, model_id, type(model_id))
                        model_ids.append(model_id)
                model_id = str(max(model_ids))
                model_fname = os.path.join(output_dir, "model_" + model_id + ".pt")
                with open(model_fname, "rb") as f:
                    print(f"Loading model: {model_fname}")
                    model = torch.load(f, map_location=device)
            else:
                raise ValueError(f"Model does not exist: {model_fname}")
        else:
            with open(model_fname, "rb") as f:
                print(f"Loading model: {model_fname}")
                model = torch.load(f, map_location=device)

        return model
    

    def load_model_torchscript(self, params, model_id, device_type="cpu"):
        output_dir = os.path.join(params["mlflow_io"]["output_dir"], "model")
        device = torch.device(device_type)
        model_fname = os.path.join(output_dir, "model_" + model_id + "_torchscript_" + device_type + ".pt")
        if not os.path.exists(model_fname):
            raise ValueError(f"Model torchscript does not exist: {model_fname}")
        else:
            with open(model_fname, "rb") as f:
                print(f"Loading model torchscript: {model_fname}")
                model = torch.jit.load(f, map_location=device)

        return model
    

    def save_prediction_results(self, params, dicts, fname):
        output_dir = os.path.join(params["mlflow_io"]["output_dir"], "prediction")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_fname = os.path.join(output_dir, fname + ".pt")
        with open(output_fname, "wb") as f:
            torch.save(dicts, f)
            print(f"\nSucessfully saved final prediction results: {output_fname}")