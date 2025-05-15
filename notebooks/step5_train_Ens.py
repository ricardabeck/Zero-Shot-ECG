"""
To run in background:

    conda activate torch_dpl
    cd notebooks/
    nohup python -m step5_train_Ens > logs/step5_all.log 2>&1
    nohup python -m step5_train_Ens > logs/step5_all_more.log 2>&1
    nohup python -m step5_train_Ens > logs/step5_all_trueDP.log 2>&1

Runtime:
    ~ 1 minute per epsilon

"""

# Generic libraries
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import scipy as sp
import torch
from torch import nn
from scipy import io as sio
from scipy import signal as sps
from scipy import linalg as spl
from os.path import join as osj
from sklearn.pipeline import Pipeline
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
from ecg_utilities import get_performance_metrics

# Pytorch libraries
import torch.nn.functional as Func
from pytorch_sklearn import NeuralNetwork
from pytorch_sklearn.callbacks import WeightCheckpoint, Verbose, LossPlot, EarlyStopping, Callback, CallbackInfo
from pytorch_sklearn.utils.func_utils import to_safe_tensor


# FUNCTIONS
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

def load_dictionary(m, epsilon, patient_id):
    """
    Reads the pickled ECG dictionary from the given path for the given patient.
    """
    with open(osj("..", "dp_data_single", "dictionaries", f"{m}_dictionary.pkl"), "rb") as f:
        dict_all = pickle.load(f)
        D = dict_all[epsilon][patient_id]
        del dict_all
        F = spl.null_space(D.T)
        F = spl.orth(F).T
    return D, F

def load_net(mechanism, e):
    e = str(e)
    mechanism = str(mechanism)

    with open(osj("..", "dp_models", "DA", mechanism, f"{e}_nets.pkl"), "rb") as f:
        net_e = pickle.load(f)
    return net_e

def extract_array_of_dict_of_confmat(arr):
    return np.stack([np.stack(list(d.values())) for d in arr])

def extract_array_of_dict_of_dict_of_confmat(arr):
    return np.stack([np.stack([np.stack(list(d2.values())) for d2 in d1.values()]) for d1 in arr])

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

def save_results_e(model, mechanism, e, p, c, n): 
        
    # save performance
    with open(osj("..", "dp_models", model, mechanism, f"{e}_performance.pkl"), "rb") as f:
        pickle.dump(p, f) 
            
    # save config
    with open(osj("..", "dp_models", model, mechanism, f"{e}_config.pkl"), "rb") as f: 
        pickle.dump(c, f) 
        
    # save model
    with open(osj("..", "dp_models", model, mechanism, f"{e}_nets.pkl"), "rb") as f:
        pickle.dump(n, f) 

def save_results_ens(model, mechanism, e, perf, cms, config):
    # save performance
    with open(osj("..", "dp_models", model, mechanism, f"{e}_performance.pkl"), "wb") as f:
        pickle.dump(perf, f) 

    # save cms
    with open(osj("..", "dp_models", model, mechanism, f"{e}_cms.pkl"), "wb") as f: 
        pickle.dump(cms, f) 
            
    # save config
    with open(osj("..", "dp_models", model, mechanism, f"{e}_config.pkl"), "wb") as f: 
        pickle.dump(config, f) 

def save_results_ens_val(model, mechanism, e, perf, cms, config, confs):
    # save performance
    with open(osj("..", "dp_models", model, mechanism, f"{e}_performance.pkl"), "wb") as f:
        pickle.dump(perf, f) 

    # save cms
    with open(osj("..", "dp_models", model, mechanism, f"{e}_cms.pkl"), "wb") as f: 
        pickle.dump(cms, f) 
            
    # save config
    with open(osj("..", "dp_models", model, mechanism, f"{e}_config.pkl"), "wb") as f: 
        pickle.dump(config, f) 
        
    # save confs
    with open(osj("..", "dp_models", model, mechanism, f"{e}_confs.pkl"), "wb") as f:
        pickle.dump(confs, f)
        


def train_ens_model(): 
    # p_method = ["laplace", "bounded_n", "gaussian_a", "laplace_truedp"]
    p_method = ["laplace_truedp"]
    model = "DA_Ens" # training DA Ensemble model
    # hp_epsilon_values = [0.11, 0.21, 0.31, 0.41, 0.61, 0.71, 0.81, 1.11, 1.21, 1.31, 1.41]
    valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)

    max_epochs = [-1]
    batch_sizes = [1024]
    confidences = [0, *np.linspace(0.5, 1, 51)]
    repeats = 10


    ########  MECHANISM  ########
    for mechanism in p_method:
        logger.info(f"Setup {mechanism} for ensemble model ...")

        if mechanism == "bounded_n":
            hp_epsilon_values = [0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.91,
                                2.01, 2.51, 2.91,
                                3.01, 3.51, 3.91,
                                4.01, 4.51, 4.91,
                                5.01, 5.51, 5.91,
                                6.01, 6.51, 6.91,
                                7.01, 7.51, 7.91,
                                8.01, 8.51, 8.91,
                                9.01, 9.51, 9.91, 10]
        elif mechanism == "laplace_truedp":
            hp_epsilon_values = [0.00001, 0.0001, 0.001, 
                                0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91]
        else:
            hp_epsilon_values = [0.00001, 0.0001, 0.001, 
                                0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.91,
                                2.01, 2.51, 2.91,
                                3.01, 3.51, 3.91,
                                4.01, 4.51, 4.91,
                                5.01, 5.51, 5.91,
                                6.01, 6.51, 6.91,
                                7.01, 7.51, 7.91,
                                8.01, 8.51, 8.91,
                                9.01, 9.51, 9.91, 10]
            
        ########  EPSILON  ########
        for epsilon in hp_epsilon_values:

            if os.path.exists(osj("..", "dp_models", model, mechanism, f"{epsilon}_performance.pkl")):
                logger.info(f"Skipping existing {epsilon} ...")
                
            else:

                if os.path.exists(osj("..", "dp_models", "DA", mechanism, f"{epsilon}_nets.pkl")):

                    single_dataset, trio_dataset = load_domain_data_epsilon(mechanism, epsilon)
                    logger.info(f"Starting with epsilon {epsilon} ...")

                    epsilon_patient_cms = []
                    epsilon_cms = []
                    epsilon_performance = {}

                    nets_all_e = load_net(mechanism, epsilon)

                    ########  REPEATS  ########
                    for repeat in range(repeats):
                        logger.info(f"Repeat {repeat+1}/10 for epsilon {epsilon}")

                        patient_cms = {conf:{} for conf in confidences}
                        cm = {conf:torch.zeros(2, 2) for conf in confidences}

                        repeat_patient_cms = {}
                        
                        for i, patient_id in enumerate(valid_patients):
                            # logger.info(f"Training for patient {patient_id} ...")
                            dataset = load_N_channel_dataset(single_dataset[patient_id], trio_dataset[patient_id])
                            train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids = dataset.values()
                            
                            # For consulting through error energy.
                            D, F = load_dictionary(mechanism, epsilon, patient_id)
                            D, F = torch.Tensor(D), torch.Tensor(F)

                            ## Consulting Exponential - Gaussian.
                            BF = BayesianFit()
                            EF = ExponentialFit()
                            GF = GaussianFit()

                            # Train error.
                            train_E, E_healthy, E_arrhyth = get_error_one_patient(train_X[:, 0, :].squeeze(), F, y=train_y, as_energy=True)
                            # _, E_healthy, E_arrhyth = get_error_per_patient(train_X[:, 0, :].squeeze(), ids=train_ids, DICT_PATH=DICT_PATH, y=train_y, as_energy=True)
                            
                            EF.fit(E_healthy)
                            GF.fit(E_arrhyth)
                            consult_train_y = torch.Tensor(BF.predict(train_E, EF, GF) <= 0.5).long()
                            
                            # Test Error (be careful, we check (<= 0.5) because EF => healthy => label 0)
                            test_E = get_error_one_patient(test_X[:, 0, :].squeeze(), F, as_energy=True)
                            
                            EF.fit(E_healthy)
                            GF.fit(E_arrhyth)
                            consult_test_y = torch.Tensor(BF.predict(test_E, EF, GF) <= 0.5).long()
                            ##

                            # Load the neural network.
                            model_base = get_base_model(in_channels=train_X.shape[1])
                            model_base = model_base.to("cuda")
                            crit = nn.CrossEntropyLoss()
                            optim = torch.optim.AdamW(params=model_base.parameters())
                            
                            net = NeuralNetwork.load_class_from_data(nets_all_e[repeat][patient_id], model_base, optim, crit)
                            weight_checkpoint_val_loss = net.cbmanager.callbacks[1]  # <- this needs to change in case weight checkpoint is not the second callback.
                            net.load_weights(weight_checkpoint_val_loss)
                            
                            # Test predictions and probabilities.
                            pred_y = net.predict(test_X, batch_size=1024, use_cuda=True, fits_gpu=True, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
                            prob_y = net.predict_proba(test_X).cpu()
                            softmax_prob_y = Func.softmax(prob_y, dim=1).max(dim=1).values
                            
                            for conf in confidences:
                                low_confidence = softmax_prob_y < conf
                                high_confidence = softmax_prob_y >= conf

                                final_pred_y = torch.Tensor(np.select([low_confidence, high_confidence], [consult_test_y, pred_y])).long()
                                cm_exp = get_confusion_matrix(final_pred_y, test_y, pos_is_zero=False)

                                patient_cms[conf][patient_id] = cm_exp
                                cm[conf] += cm_exp
                                
                            
                        epsilon_patient_cms.append(repeat_patient_cms)
                        epsilon_cms.append(cm)


                    config = dict(
                        learning_rate=0.001,
                        max_epochs=max_epochs[0],
                        batch_size=batch_sizes[0],
                        optimizer=optim.__class__.__name__,
                        loss=crit.__class__.__name__,
                        early_stopping="true",
                        checkpoint_on=weight_checkpoint_val_loss.tracked,
                        dataset="default+trio",
                        info="Results replicated for GitHub, DA + Ensemble + All C."
                    )

                    epsilon_cms = extract_array_of_dict_of_confmat(epsilon_cms)
                    
                    epsilon_performance["DA"]  = get_performance_metrics(epsilon_cms.sum(axis=0)[0])  # only DA
                    epsilon_performance["NPE"] = get_performance_metrics(epsilon_cms.sum(axis=0)[-1]) # only NPE
                    epsilon_performance["ENS"] = get_performance_metrics(epsilon_cms[:, 1:-1, :, :].sum(axis=(0, 1))) # avg over quantized C
                    
                    logger.info(f"All repeats for epsilon {epsilon} done. Saving now ...")
                    save_results_ens(model, mechanism, epsilon, epsilon_performance, epsilon_cms, config)
                
                else:

                    logger.info(f"Epsilon {epsilon} was not prepared yet.")


def train_ens_val_model():

    # p_method = ["laplace", "bounded_n", "gaussian_a", "laplace_truedp"]
    p_method = ["laplace_truedp"]
    model = "Ens_val" # training Ensemble validation model

    valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)

    max_epochs = [-1]
    batch_sizes = [1024]
    confidences = [0, *np.linspace(0.5, 1, 51)]
    repeats = 10


    ########  MECHANISM  ########
    for mechanism in p_method:
        logger.info(f"Setup {mechanism} for ensemble validation ...")

        if mechanism == "bounded_n":
            hp_epsilon_values = [0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.91,
                                2.01, 2.51, 2.91,
                                3.01, 3.51, 3.91,
                                4.01, 4.51, 4.91,
                                5.01, 5.51, 5.91,
                                6.01, 6.51, 6.91,
                                7.01, 7.51, 7.91,
                                8.01, 8.51, 8.91,
                                9.01, 9.51, 9.91, 10]
        elif mechanism == "laplace_truedp":
            hp_epsilon_values = [0.00001, 0.0001, 0.001, 
                                0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91]
        else:
            hp_epsilon_values = [0.00001, 0.0001, 0.001, 
                                0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.91,
                                2.01, 2.51, 2.91,
                                3.01, 3.51, 3.91,
                                4.01, 4.51, 4.91,
                                5.01, 5.51, 5.91,
                                6.01, 6.51, 6.91,
                                7.01, 7.51, 7.91,
                                8.01, 8.51, 8.91,
                                9.01, 9.51, 9.91, 10]
            
        ########  EPSILON  ########
        for epsilon in hp_epsilon_values:

            if os.path.exists(osj("..", "dp_models", model, mechanism, f"{epsilon}_performance.pkl")):
                logger.info(f"Skipping existing {epsilon} ...")
                
            else:
                    
                if os.path.exists(osj("..", "dp_models", "DA", mechanism, f"{epsilon}_nets.pkl")):

                    single_dataset, trio_dataset = load_domain_data_epsilon(mechanism, epsilon)
                    logger.info(f"Starting with epsilon {epsilon} ...")

                    epsilon_patient_cms = []
                    epsilon_cms = []
                    epsilon_performance = {}
                    epsilon_confs = []

                    nets_all_e = load_net(mechanism, epsilon)

                    ########  REPEATS  ########
                    for repeat in range(repeats):
                        logger.info(f"Repeat {repeat+1}/10 for epsilon {epsilon}")

                        repeat_patient_cms = {}
                        repeat_confs = []
                        cm = torch.zeros(2, 2)
                        
                        for i, patient_id in enumerate(valid_patients):
                            # logger.info(f"Training for patient {patient_id} ...")
                            dataset = load_N_channel_dataset(single_dataset[patient_id], trio_dataset[patient_id])
                            train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids = dataset.values()
                            
                            # For consulting through error energy.
                            D, F = load_dictionary(mechanism, epsilon, patient_id)
                            D, F = torch.Tensor(D), torch.Tensor(F)

                            ## Consulting Exponential - Gaussian.
                            BF = BayesianFit()
                            EF = ExponentialFit()
                            GF = GaussianFit()

                            # Train error.
                            train_E, E_healthy, E_arrhyth = get_error_one_patient(train_X[:, 0, :].squeeze(), F, y=train_y, as_energy=True)
                            # _, E_healthy, E_arrhyth = get_error_per_patient(train_X[:, 0, :].squeeze(), ids=train_ids, DICT_PATH=DICT_PATH, y=train_y, as_energy=True)
                            
                            EF.fit(E_healthy)
                            GF.fit(E_arrhyth)
                            consult_train_y = torch.Tensor(BF.predict(train_E, EF, GF) <= 0.5).long()
                                    
                            # Validation error.
                            val_E, val_E_healthy, val_E_arrhyth = get_error_one_patient(val_X[:, 0, :].squeeze(), F, y=val_y, as_energy=True)
                            
                            EF.fit(val_E_healthy)
                            GF.fit(val_E_arrhyth)
                            consult_val_y = torch.Tensor(BF.predict(val_E, EF, GF) <= 0.5).long()
                            
                            # Test Error (be careful, we check (<= 0.5) because EF => healthy => label 0)
                            test_E = get_error_one_patient(test_X[:, 0, :].squeeze(), F, as_energy=True)
                            
                            EF.fit(E_healthy)
                            GF.fit(E_arrhyth)
                            consult_test_y = torch.Tensor(BF.predict(test_E, EF, GF) <= 0.5).long()
                            ##

                            # Load the neural network.
                            model_base = get_base_model(in_channels=train_X.shape[1])
                            model_base = model_base.to("cuda")
                            crit = nn.CrossEntropyLoss()
                            optim = torch.optim.AdamW(params=model_base.parameters())
                            
                            net = NeuralNetwork.load_class_from_data(nets_all_e[repeat][patient_id], model_base, optim, crit)
                            weight_checkpoint_val_loss = net.cbmanager.callbacks[1]  # <- this needs to change in case weight checkpoint is not the second callback.
                            net.load_weights(weight_checkpoint_val_loss)
                            
                            # Validation predictions and probabilities.
                            pred_y = net.predict(val_X, batch_size=1024, use_cuda=True, fits_gpu=True, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
                            prob_y = net.predict_proba(val_X).cpu()
                            softmax_prob_y = Func.softmax(prob_y, dim=1).max(dim=1).values
                            
                            # Choose consult threshold from validation set that maximizes F1.
                            maxF1 = float("-inf")
                            secondMaxF1 = float("-inf")
                            maxConf = -1
                            secondMaxConf = -1
                            
                            for conf in confidences:
                                low_confidence = softmax_prob_y < conf
                                high_confidence = softmax_prob_y >= conf

                                final_pred_y = torch.Tensor(np.select([low_confidence, high_confidence], [consult_val_y, pred_y])).long()
                                val_cm = get_confusion_matrix(final_pred_y, val_y, pos_is_zero=False)
                                
                                f1 = get_performance_metrics(val_cm)["f1"]
                                f1 = np.nan_to_num(f1)
                                
                                if f1 >= (maxF1 - 1e-3):
                                    secondMaxF1 = maxF1
                                    secondMaxConf = maxConf
                                    maxF1 = f1
                                    maxConf = conf
                            
                            repeat_confs.append(maxConf)
                            
                            # Test predictions and probabilities.
                            pred_y = net.predict(test_X, batch_size=1024, use_cuda=True, fits_gpu=True, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
                            prob_y = net.predict_proba(test_X).cpu()
                            softmax_prob_y = Func.softmax(prob_y, dim=1).max(dim=1).values
                            
                            # Use the confidence chosen above instead of iterating all confidences.
                            for conf in [maxConf]:
                                low_confidence = softmax_prob_y < conf
                                high_confidence = softmax_prob_y >= conf

                                final_pred_y = torch.Tensor(np.select([low_confidence, high_confidence], [consult_test_y, pred_y])).long()
                                cm_exp = get_confusion_matrix(final_pred_y, test_y, pos_is_zero=False)

                                repeat_patient_cms[patient_id] = cm_exp
                                cm += cm_exp
                                
                            #print_progress(i + 1, len(valid_patients), opt=[f"{patient_id}"])
                            
                        epsilon_patient_cms.append(repeat_patient_cms)
                        epsilon_cms.append(cm)
                        epsilon_confs.append(repeat_confs)


                    config = dict(
                        learning_rate=0.001,
                        max_epochs=max_epochs[0],
                        batch_size=batch_sizes[0],
                        optimizer=optim.__class__.__name__,
                        loss=crit.__class__.__name__,
                        early_stopping="true",
                        checkpoint_on=weight_checkpoint_val_loss.tracked,
                        dataset="default+trio",
                        info="Results replicated for GitHub, DA + Ensemble + Validation C."
                    )

                    epsilon_performance = get_performance_metrics(torch.stack(epsilon_cms).sum(dim=0))
                    
                    logger.info(f"All repeats for epsilon {epsilon} done. Saving now ...")
                    save_results_ens_val(model, mechanism, epsilon, epsilon_performance, epsilon_cms, config, epsilon_confs)

                else:

                    logger.info(f"Epsilon {epsilon} was not prepared yet.")


def main():
    #train_ens_model()
    train_ens_val_model()

if __name__ == "__main__":
    main()