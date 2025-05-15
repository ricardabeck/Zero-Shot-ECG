
"""
To run in background

    conda activate torch_dpl
    cd notebooks/
    nohup python -m step6a_shadow_data > logs/step6_shadow_data.log 2>&1

Runtime:
    ~ 3 mins to load the data per mechanism
    1 second per epsilon
"""

# Generic libraries
import numpy as np
import pandas as pd
from os.path import join as osj
import pickle
import os
import torch

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

# Paper Libraries for functions
from ecg_utilities import *

# Pytorch libraries
import torch.nn.functional as Func
from pytorch_sklearn import NeuralNetwork
from pytorch_sklearn.callbacks import WeightCheckpoint, Verbose, LossPlot, EarlyStopping, Callback, CallbackInfo
from pytorch_sklearn.utils.func_utils import to_safe_tensor


def load_domain_data_epsilon(m): 
    with open(osj("..", f"dp_data_single", "dataset_training", f"{m}_domain_adapted.pkl"), "rb") as f:
        single = pickle.load(f)
    with open(osj("..", f"dp_data_trio", "dataset_training", f"{m}_domain_adapted.pkl"), "rb") as f:
        trio = pickle.load(f)
    return single, trio  

def load_N_channel_dataset(single_data, trio_data):
    """
    Loads the ECG dataset at the given path(s) for one patient at a time. Each dataset will be added as a new
    channel in the given order.
    """
    default_dataset = dataset_to_tensors(single_data)
    dataset_other   = dataset_to_tensors(trio_data)
    default_dataset = add_dataset(default_dataset, dataset_other)
    
    return default_dataset

def load_real_2channel_dataset(patient_id):
    """
    Loads the ECG dataset at the given path(s) for one patient at a time. Each dataset will be added as a new
    channel in the given order.
    """
    with open(osj("..", f"data_single", "dataset_training", "domain_adapted", f"patient_{patient_id}_dataset.pkl"), "rb") as f:
        single_data = pickle.load(f)
    with open(osj("..", f"data_trio", "dataset_training", "domain_adapted", f"patient_{patient_id}_dataset.pkl"), "rb") as f:
        trio_data = pickle.load(f)

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

def save_train_test_data(m, e, train_test_data):
    with open(osj("..", "dp_models", "train_test_data", m, f"{e}_data.pkl"), "wb") as f:
        pickle.dump(train_test_data, f) 

def save_real_data(train_test_data):
    with open(osj("..", "dp_models", "train_test_data", "real_data.pkl"), "wb") as f:
        pickle.dump(train_test_data, f) 

# assumes partial leakage / availability of the data
def get_train_test_data():

    p_method = ["laplace", "bounded_n", "gaussian_a", "laplace_truedp"]
    attack_setup = pd.read_pickle("../files/attack_setup.pkl") # dataframe with columns: Model, Method, Epsilon, Metric, Value

    valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)

    last_mechanism = None

    ########  MECHANISM  ########
    for mechanism in p_method:

        hp_epsilon_values = list(set(attack_setup[attack_setup["Method"] == mechanism]["Epsilon"].tolist()))
        
        ########  EPSILON  ########
        for epsilon in hp_epsilon_values:
            epsilon_ids = {}

            if os.path.exists(osj("..", "dp_models", "train_test_data", mechanism, f"{epsilon}_data.pkl")):
                logger.info(f"Skipping existing epsilon {epsilon} for {mechanism} ...")

            else:
                
                if mechanism != last_mechanism:
                    logger.info(f"Getting data for epsilon {mechanism} ...")
                    single_dataset, trio_dataset = load_domain_data_epsilon(mechanism)
                    last_mechanism = mechanism
                
                ########  PATIENTS  ########
                for i, patient_id in enumerate(valid_patients):
                    patient_ids = {}
                    
                    dataset = load_N_channel_dataset(single_dataset[epsilon][patient_id], trio_dataset[epsilon][patient_id])
                    train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids = dataset.values()

                    patient_ids["Train_x"] = train_X
                    patient_ids["Train_y"] = train_y
                    patient_ids["Train"]   = train_ids # used for training the real model
                    patient_ids["Val_x"]   = val_X
                    patient_ids["Val_y"]   = val_y
                    patient_ids["Val"]     = val_ids   # used for training the real model - assume was leaked
                    patient_ids["Test_x"]  = test_X  
                    patient_ids["Test_y"]  = test_y  
                    patient_ids["Test"]    = test_ids  # used for testing the real model - assume is public

                    epsilon_ids[patient_id] = patient_ids
            
                save_train_test_data(mechanism, epsilon, epsilon_ids)
                logger.info(f"Saved data for epsilon {epsilon}.")


def get_real_train_test_data():

    valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)
    train_test_data = {}
    mechanism = ["no_dp"]

    ########  PATIENTS  ########
    for i, patient_id in enumerate(valid_patients):
        patient_ids = {}
                    
        dataset = load_real_2channel_dataset(patient_id)
        train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids = dataset.values()

        patient_ids["Train_x"] = train_X
        patient_ids["Train_y"] = train_y
        patient_ids["Train"]   = train_ids # used for training the real model
        patient_ids["Val_x"]   = val_X
        patient_ids["Val_y"]   = val_y
        patient_ids["Val"]     = val_ids   # used for training the real model - assume was leaked
        patient_ids["Test_x"]  = test_X  
        patient_ids["Test_y"]  = test_y  
        patient_ids["Test"]    = test_ids  # used for testing the real model - assume is public

        train_test_data[patient_id] = patient_ids

    save_real_data(train_test_data)
    logger.info(f"Saved real train-test data.")

def main():

    logger.info("Retrieving train-test data for shadow model training")
    get_train_test_data()
    #get_real_train_test_data() 

if __name__ == "__main__":
    main()
