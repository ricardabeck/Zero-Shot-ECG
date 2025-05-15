
"""
To run in background:

    conda activate torch_dpl
    cd notebooks/
    nohup python -m step4_train_DA > logs/step4_all.log 2>&1
    nohup python -m step4_train_DA > logs/step4_all_trueDP.log 2>&1

Runtime:
    ~ 37 hours per mechanism
    ~ 25 minutes - 1 hour per epsilon

"""

# Generic libraries
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import scipy as sp
from scipy import io as sio
from scipy import signal as sps
from scipy import linalg as spl
from os.path import join as osj
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import wfdb
import pickle
import copy
import random
import import_ipynb
import os
import sys
import json
from bisect import bisect
from collections import defaultdict 

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

# Paper Libraries for functions
from ecg_utilities import *
from progress_bar import print_progress
from ecg_utilities import get_performance_metrics

# Pytorch libraries
import torch.nn.functional as Func
from pytorch_sklearn import NeuralNetwork
from pytorch_sklearn.callbacks import WeightCheckpoint, Verbose, LossPlot, EarlyStopping, Callback, CallbackInfo
from pytorch_sklearn.utils.func_utils import to_safe_tensor


def load_domain_data_epsilon(m,e): 
    with open(osj("..", f"dp_data_single", "dataset_training", f"{m}_domain_adapted.pkl"), "rb") as f:
        single = pickle.load(f)
        single_epsilon = single[e]
        del single
    with open(osj("..", f"dp_data_trio", "dataset_training", f"{m}_domain_adapted.pkl"), "rb") as f:
        trio = pickle.load(f)
        trio_epsilon = trio[e]
        del trio
    return single_epsilon, trio_epsilon  


def load_N_channel_dataset(single_data, trio_data):
    """
    Loads the ECG dataset at the given path(s) for one patient at a time. Each dataset will be added as a new
    channel in the given order.
    """
    default_dataset = dataset_to_tensors(single_data)
    dataset_other   = dataset_to_tensors(trio_data)
    default_dataset = add_dataset(default_dataset, dataset_other)
    
    return default_dataset

def dataset_to_tensors(dataset):
    """
    Converts the given dataset to torch tensors in appropriate data types and shapes.
    """
    dataset = dataset.copy()
    train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids = dataset.values()
    dataset["train_X"] = torch.Tensor(train_X).float().reshape(-1, 1, train_X.shape[1])
    dataset["train_y"] = torch.Tensor(train_y).long()
    dataset["train_ids"] = torch.Tensor(train_ids).long()
    dataset["val_X"] = torch.Tensor(val_X).float().reshape(-1, 1, val_X.shape[1])
    dataset["val_y"] = torch.Tensor(val_y).long()
    dataset["val_ids"] = torch.Tensor(val_ids).long()
    dataset["test_X"] = torch.Tensor(test_X).float().reshape(-1, 1, test_X.shape[1])
    dataset["test_y"] = torch.Tensor(test_y).long()
    dataset["test_ids"] = torch.Tensor(test_ids).long()
    return dataset

def add_dataset(dataset, dataset_other):
    """
    Adds another dataset to an already existing one, increasing the number of channels.
    """
    dataset = dataset.copy()
    
    assert torch.equal(dataset["train_y"], dataset_other["train_y"]), "Training ground truths are different. Possibly shuffled differently."
    assert torch.equal(dataset["val_y"], dataset_other["val_y"]), "Validation ground truths are different. Possibly shuffled differently."
    assert torch.equal(dataset["test_y"], dataset_other["test_y"]), "Test ground truths are different. Possibly shuffled differently."
    
    train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids = dataset.values()
    train_other_X, _, _, val_other_X, _, _, test_other_X, _, _ = dataset_other.values()
    dataset["train_X"] = torch.cat((train_X, train_other_X), dim=1)
    dataset["val_X"] = torch.cat((val_X, val_other_X), dim=1)
    dataset["test_X"] = torch.cat((test_X, test_other_X), dim=1)
    
    return dataset

def get_base_model(in_channels):
    """
    Returns the model from paper: Personalized Monitoring and Advance Warning System for Cardiac Arrhythmias.
    """
    # Input size: 128x1
    # 128x1 -> 122x32 -> 40x32 -> 34x16 -> 11x16 -> 5x16 -> 1x16
    model = nn.Sequential(
        nn.Conv1d(in_channels, 32, kernel_size=7, padding=0, bias=True),
        nn.MaxPool1d(3),
        nn.Tanh(),
        
        nn.Conv1d(32, 16, kernel_size=7, padding=0, bias=True),
        nn.MaxPool1d(3),
        nn.Tanh(),
        
        nn.Conv1d(16, 16, kernel_size=7, padding=0, bias=True),
        nn.MaxPool1d(3),
        nn.Tanh(),
        
        nn.Flatten(),
        
        nn.Linear(16, 32, bias=True),
        nn.ReLU(),
        
        nn.Linear(32, 2, bias=True),
    )
    return model


# model, mechanism, epsilon, performance, config, net
def save_results_e(model, mechanism, e, p, c, n): 
    e = str(e)
    mechanism = str(mechanism)
    model = str(model)
    try:
        # save performance
        with open(osj("..", "dp_models", model, mechanism, f"{e}_performance.pkl"), "wb") as f:
            pickle.dump(p, f) 
                
        # save config
        with open(osj("..", "dp_models", model, mechanism, f"{e}_config.pkl"), "wb") as f: 
            pickle.dump(c, f) 
            
        # save model
        with open(osj("..", "dp_models", model, mechanism, f"{e}_nets.pkl"), "wb") as f:
            pickle.dump(n, f) 
    except:
        logger.info(f"model: {model}, type: {type(model)}")
        logger.info(f"mechanism: {mechanism}, type: {type(mechanism)}")
        logger.info(f"e: {e}, type: {type(e)}")


def train_net():

    # p_method = ["laplace", "bounded_n", "gaussian_a", "laplace_truedp"]
    p_method = ["laplace_truedp"]
    model = "DA" # training base DA model
    
    valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)
    max_epochs = [-1]
    batch_sizes = [1024]
    repeats = 10

    ########  MECHANISM  ########
    for mechanism in p_method:
        logger.info(f"Setup for {mechanism} ...")

        if mechanism == "bounded_n":
            hp_epsilon_values = [0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091, 
                                 0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 
                                 1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.91, 
                                 2.01, 2.51, 2.91, 3.01, 3.51, 3.91, 4.01, 4.51, 4.91, 5.01, 5.51, 5.91, 6.01, 6.51, 6.91, 7.01, 7.51, 7.91, 8.01, 8.51, 8.91, 9.01, 9.51, 9.91, 10.0]
        elif mechanism == "laplace_truedp":
            hp_epsilon_values = [0.00001, 0.0001, 0.001, 
                                0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91]
        else:
            hp_epsilon_values = [0.00001, 0.0001, 
                                 0.001, 0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091, 
                                 0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 
                                 1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.91, 
                                 2.01, 2.51, 2.91, 3.01, 3.51, 3.91, 4.01, 4.51, 4.91, 5.01, 5.51, 5.91, 6.01, 6.51, 6.91, 7.01, 7.51, 7.91, 8.01, 8.51, 8.91, 9.01, 9.51, 9.91, 10.0]
                
        ########  EPSILON  ########
        for epsilon in hp_epsilon_values:

            if os.path.exists(osj("..", "dp_models", model, mechanism, f"{epsilon}_performance.pkl")):
                logger.info(f"Skipping existing {epsilon} ...")
                
            else:

                single_dataset, trio_dataset = load_domain_data_epsilon(mechanism, epsilon)

                logger.info(f"Starting with epsilon {epsilon} ...")

                epsilon_patient_cms = []
                epsilon_cms = []
                epsilon_net = {}
                epsilon_performance = {}
                
                ########  REPEATS  ########
                for repeat in range(repeats):
                    logger.info(f"Repeat {repeat+1}/10 for epsilon {epsilon}")

                    repeat_patient_cms = {}
                    repeat_net = {}

                    cm = torch.zeros(2, 2)
                    
                    for i, patient_id in enumerate(valid_patients):
                        logger.info(f"Training for patient {patient_id} ...")

                        patient_net = {}
                        dataset = load_N_channel_dataset(single_dataset[patient_id], trio_dataset[patient_id])
                        train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids = dataset.values()

                        # Neural Network setup
                        model_base = get_base_model(in_channels=train_X.shape[1])
                        crit = nn.CrossEntropyLoss()
                        optim = torch.optim.AdamW(params=model_base.parameters())
                        net = NeuralNetwork(model_base, optim, crit)
                        weight_checkpoint_val_loss = WeightCheckpoint(tracked="val_loss", mode="min")
                        early_stopping = EarlyStopping(tracked="val_loss", mode="min", patience=15)

                        # Neural Network training
                        net.fit(
                            train_X=train_X,
                            train_y=train_y,
                            validate=True,
                            val_X=val_X,
                            val_y=val_y,
                            max_epochs=max_epochs[0],
                            batch_size=batch_sizes[0],
                            use_cuda=True,
                            fits_gpu=True,
                            callbacks=[weight_checkpoint_val_loss, early_stopping],
                        )
                        
                        net.load_weights(weight_checkpoint_val_loss)
                        pred_y = net.predict(test_X, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
                        
                        # In order to save the trained weights
                        patient_net = NeuralNetwork.return_class(net) # save best model for this patient, in this repeat
                        repeat_net[patient_id] = patient_net

                        cur_cm = get_confusion_matrix(pred_y, test_y, pos_is_zero=False)
                        repeat_patient_cms[patient_id] = cur_cm # patient_cm
                        cm += cur_cm # 

                    epsilon_net[repeat] = repeat_net

                    epsilon_patient_cms.append(repeat_patient_cms)
                    epsilon_cms.append(cm.detach().clone())  # Konvertiert in eine Python-Liste

                epsilon_cms = np.stack(epsilon_cms).astype(int)
                epsilon_performance = get_performance_metrics(epsilon_cms.sum(axis=0))

                # after all repeats (per epsilon)
                config = dict(
                    learning_rate=0.001,
                    max_epochs=max_epochs[0],
                    batch_size=batch_sizes[0],
                    optimizer=optim.__class__.__name__,
                    loss=crit.__class__.__name__,
                    early_stopping="true",
                    checkpoint_on=weight_checkpoint_val_loss.tracked,
                    dataset="default+trio",
                    info="2-channel run, domain adapted, consulting with default dictionary, and trying all thresholds, saves weights"
                )

                logger.info(f"All repeats for epsilon {epsilon} done. Saving now ...")
                save_results_e(model, mechanism, epsilon, epsilon_performance, config, epsilon_net)

def main():
    train_net()

if __name__ == "__main__":
    main()