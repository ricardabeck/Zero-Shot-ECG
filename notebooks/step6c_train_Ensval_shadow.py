"""
To run:

    conda activate torch_dpl
    cd notebooks/
    nohup python -m step6c_train_Ensval_shadow > logs/step6_Ensval_shadow_dp.log 2>&1
    nohup python -m step6c_train_Ensval_shadow > logs/step6_Ensval_shadow_real.log 2>&1

Runtime:
    ~ 1 minute per epsilon
    
"""

# Generic libraries
import numpy as np
import pandas as pd
import sklearn as sk
import scipy as sp
from scipy import io as sio
from scipy import signal as sps
from scipy import linalg as spl
from os.path import join as osj
from sklearn.pipeline import Pipeline
import pickle
import copy
import random
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

# Pytorch libraries
import torch.nn.functional as Func
from pytorch_sklearn import NeuralNetwork
from pytorch_sklearn.callbacks import WeightCheckpoint, Verbose, LossPlot, EarlyStopping, Callback, CallbackInfo
from pytorch_sklearn.utils.func_utils import to_safe_tensor
from sklearn.model_selection import train_test_split

def load_net(mechanism, e):
    with open(osj("..", "dp_models", "DA_shadow", mechanism, f"{e}_nets.pkl"), "rb") as f:
        net_e = pickle.load(f)
    return net_e

def load_net_real(mechanism):
    with open(osj("..", "dp_models", "DA_shadow", mechanism, "nets.pkl"), "rb") as f:
        net_e = pickle.load(f)
    return net_e

def load_train_test_data(m, e):
    with open(osj("..", "dp_models", "train_test_data", m, f"{e}_data.pkl"), "rb") as f:
        data = pickle.load(f)
    return data 

def load_real_train_test_data():
    with open(osj("..", "dp_models", "train_test_data", "real_data.pkl"), "rb") as f:
        data = pickle.load(f)
    return data 

def load_dictionary_dp(m, epsilon, patient_id):
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

def extract_array_of_dict_of_confmat(arr):
    return np.stack([np.stack(list(d.values())) for d in arr])

def extract_array_of_dict_of_dict_of_confmat(arr):
    return np.stack([np.stack([np.stack(list(d2.values())) for d2 in d1.values()]) for d1 in arr])

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

def save_results_ens(model, mechanism, e, prob, perf, config):
    # save performance
    with open(osj("..", "dp_models", model, mechanism, f"{e}_performance.pkl"), "wb") as f:
        pickle.dump(perf, f) 

    # save probabilities
    with open(osj("..", "dp_models", model, mechanism, f"{e}_probs.pkl"), "wb") as f:
        pickle.dump(prob, f) 
            
    # save config
    with open(osj("..", "dp_models", model, mechanism, f"{e}_config.pkl"), "wb") as f: 
        pickle.dump(config, f) 

def save_results_ens_real(model, mechanism, prob, perf, config):
    # save performance
    with open(osj("..", "dp_models", model, mechanism, "performance.pkl"), "wb") as f:
        pickle.dump(perf, f) 

    # save probabilities
    with open(osj("..", "dp_models", model, mechanism, "probs.pkl"), "wb") as f:
        pickle.dump(prob, f) 
            
    # save config
    with open(osj("..", "dp_models", model, mechanism, "config.pkl"), "wb") as f: 
        pickle.dump(config, f) 
        


###### Shadow Training ######
def train_ens_val_shadow_model():

    p_method = ["laplace", "bounded_n", "gaussian_a", "laplace_truedp"]
    model = "Ens_val_shadow" 
    valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)

    attack_setup = pd.read_pickle("../files/attack_setup.pkl") # dataframe with columns: Model, Method, Epsilon, Metric, Value   
    attack_setup_val = attack_setup[attack_setup["Model"] == "Ens_val"] 

    max_epochs = [100]
    batch_sizes = [1024]
    confidences = [0, *np.linspace(0.5, 1, 51)]
    repeats = 10
  
    ########  MECHANISM  ########
    for mechanism in p_method:
        logger.info(f"Setup {mechanism} for shadow ensemble model ...")
        hp_epsilon_values = attack_setup_val[attack_setup_val["Method"] == mechanism]["Epsilon"].tolist()

        ########  EPSILON  ########
        for epsilon in hp_epsilon_values:
            
            if os.path.exists(osj("..", "dp_models", model, mechanism, f"{epsilon}_performance.pkl")):
                logger.info(f"Skipping existing {epsilon} ...")
                
            else:

                if os.path.exists(osj("..", "dp_models", "DA_shadow", mechanism, f"{epsilon}_nets.pkl")):
                    logger.info(f"Pretrained net available for epsilon {epsilon}")
                    net_e = load_net(mechanism, epsilon)  

                    train_test_data = load_train_test_data(mechanism, epsilon)
                    epsilon_patient_cms = []
                    epsilon_cms = []
                    epsilon_confs = []
                    epsilon_performance = {}
                    epsilon_prob = dict.fromkeys(valid_patients)

                    ########  REPEATS  ########
                    for repeat in range(repeats):
                        logger.info(f"Repeat {repeat+1}/10 for epsilon {epsilon}")

                        patient_cms = {}
                        repeat_confs = []
                        cm = torch.zeros(2, 2)
                        repeat_patient_cms = {}
                            
                        for i, patient_id in enumerate(valid_patients):
                            patient_prob = dict.fromkeys(["test", "train", "val"])

                            train_X   = train_test_data[patient_id]["Val_x"]
                            train_y   = train_test_data[patient_id]["Val_y"]
                            test_X    = train_test_data[patient_id]["Test_x"]
                            test_y    = train_test_data[patient_id]["Test_y"]

                            train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42, stratify=train_y)

                            # For consulting through error energy.
                            D, F = load_dictionary_dp(mechanism, epsilon, patient_id)
                            D, F = torch.Tensor(D), torch.Tensor(F)

                            ## Consulting Exponential - Gaussian.
                            BF = BayesianFit()
                            EF = ExponentialFit()
                            GF = GaussianFit()

                            # Train error.
                            train_E, E_healthy, E_arrhyth = get_error_one_patient(train_X[:, 0, :].squeeze(), F, y=train_y, as_energy=True)

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

                            # Neural Network setup
                            model_base = get_base_model(in_channels=train_X.shape[1])
                            model_base = model_base.to("cuda")
                            crit = nn.CrossEntropyLoss()
                            optim = torch.optim.AdamW(params=model_base.parameters())

                            net = NeuralNetwork.load_class_from_data(net_e[repeat][patient_id], model_base, optim, crit)
                            weight_checkpoint_val_loss = net.cbmanager.callbacks[1]  # <- this needs to change in case weight checkpoint is not the second callback.
                            net.load_weights(weight_checkpoint_val_loss)

                            pred_y = net.predict(val_X, batch_sizes[0], use_cuda=True, fits_gpu=True, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
                            prob_y = net.predict_proba(val_X).cpu()
                            softmax_prob_val_y = Func.softmax(prob_y, dim=1).max(dim=1).values
                            # Additionally for attack model input
                            prob_train_y = net.predict_proba(train_X).cpu()
                            softmax_prob_train_y = Func.softmax(prob_train_y, dim=1).max(dim=1).values

                            # Choose consult threshold from validation set that maximizes F1.
                            maxF1 = float("-inf")
                            secondMaxF1 = float("-inf")
                            maxConf = -1
                            secondMaxConf = -1

                            for conf in confidences:
                                low_confidence = softmax_prob_val_y < conf
                                high_confidence = softmax_prob_val_y >= conf

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
                            pred_y = net.predict(test_X, batch_sizes[0], use_cuda=True, fits_gpu=True, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
                            prob_y = net.predict_proba(test_X).cpu()
                            softmax_prob_y = Func.softmax(prob_y, dim=1).max(dim=1).values
                            
                            for conf in confidences:
                                low_confidence = softmax_prob_y < conf
                                high_confidence = softmax_prob_y >= conf

                                final_pred_y = torch.Tensor(np.select([low_confidence, high_confidence], [consult_test_y, pred_y])).long()
                                cm_exp = get_confusion_matrix(final_pred_y, test_y, pos_is_zero=False)

                                patient_cms[patient_id] = cm_exp
                                cm += cm_exp

                            if repeat == 9:
                                patient_prob["test"]  = softmax_prob_y
                                patient_prob["train"] = softmax_prob_train_y
                                patient_prob["val"]   = softmax_prob_val_y
                                epsilon_prob[patient_id] = patient_prob

                        epsilon_patient_cms.append(repeat_patient_cms)
                        epsilon_cms.append(cm.detach().clone()) 
                        epsilon_confs.append(repeat_confs)
                
                    epsilon_performance = get_performance_metrics(torch.stack(epsilon_cms).sum(dim=0))

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
                        info="Shadow Model for Ensemble Validation Approach on differentially private data"
                    )

                    logger.info(f"All repeats for epsilon {epsilon} done. Saving ...")
                    save_results_ens(model, mechanism, epsilon, epsilon_prob, epsilon_performance, config)

                else:
                    
                    logger.info(f"Epsilon {epsilon} was not pretrained in base model.")
                
            

def train_ens_val_shadow_real_data():

    model = "Ens_val_shadow" 
    mechanism = "no_dp"
    valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)

    max_epochs = [100]
    batch_sizes = [1024]
    confidences = [0, *np.linspace(0.5, 1, 51)]
    repeats = 10

    all_patient_cms = []
    all_cms = []
    all_confs = []
    all_performance = {}
    all_prob = dict.fromkeys(valid_patients)

    train_test_data = load_real_train_test_data()

    if os.path.exists(osj("..", "dp_models", "DA_shadow", mechanism, "nets.pkl")):
        logger.info(f"Pretrained net available ")
        net_e = load_net_real(mechanism)  

        ########  REPEATS  ########
        for repeat in range(repeats):
            logger.info(f"Repeat {repeat+1}/10")

            patient_cms = {}
            repeat_confs = []
            cm = torch.zeros(2, 2)
            repeat_patient_cms = {}
            
            for i, patient_id in enumerate(valid_patients):
                patient_prob = dict.fromkeys(["test", "train", "val"])

                train_X   = train_test_data[patient_id]["Val_x"]
                train_y   = train_test_data[patient_id]["Val_y"]
                test_X    = train_test_data[patient_id]["Test_x"]
                test_y    = train_test_data[patient_id]["Test_y"]

                train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42, stratify=train_y)

                # For consulting through error energy.
                path = osj("..", "data_single", "dictionaries", "5min_sorted")
                D, F = load_dictionary(patient_id, path)
                D, F = torch.Tensor(D), torch.Tensor(F)

                ## Consulting Exponential - Gaussian.
                BF = BayesianFit()
                EF = ExponentialFit()
                GF = GaussianFit()

                # Train error.
                train_E, E_healthy, E_arrhyth = get_error_one_patient(train_X[:, 0, :].squeeze(), F, y=train_y, as_energy=True)

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

                # Neural Network setup
                model_base = get_base_model(in_channels=train_X.shape[1])
                model_base = model_base.to("cuda")
                crit = nn.CrossEntropyLoss()
                optim = torch.optim.AdamW(params=model_base.parameters())

                net = NeuralNetwork.load_class_from_data(net_e[repeat][patient_id], model_base, optim, crit)
                weight_checkpoint_val_loss = net.cbmanager.callbacks[1]  # <- this needs to change in case weight checkpoint is not the second callback.
                net.load_weights(weight_checkpoint_val_loss)

                pred_y = net.predict(val_X, batch_sizes[0], use_cuda=True, fits_gpu=True, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
                prob_y = net.predict_proba(val_X).cpu()
                softmax_prob_val_y = Func.softmax(prob_y, dim=1).max(dim=1).values
                # Additionally for attack model input
                prob_train_y = net.predict_proba(train_X).cpu()
                softmax_prob_train_y = Func.softmax(prob_train_y, dim=1).max(dim=1).values

                # Choose consult threshold from validation set that maximizes F1.
                maxF1 = float("-inf")
                secondMaxF1 = float("-inf")
                maxConf = -1
                secondMaxConf = -1

                for conf in confidences:
                    low_confidence = softmax_prob_val_y < conf
                    high_confidence = softmax_prob_val_y >= conf

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
                pred_y = net.predict(test_X, batch_sizes[0], use_cuda=True, fits_gpu=True, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
                prob_y = net.predict_proba(test_X).cpu()
                softmax_prob_y = Func.softmax(prob_y, dim=1).max(dim=1).values
                
                for conf in confidences:
                    low_confidence = softmax_prob_y < conf
                    high_confidence = softmax_prob_y >= conf

                    final_pred_y = torch.Tensor(np.select([low_confidence, high_confidence], [consult_test_y, pred_y])).long()
                    cm_exp = get_confusion_matrix(final_pred_y, test_y, pos_is_zero=False)

                    patient_cms[patient_id] = cm_exp
                    cm += cm_exp

                if repeat == 9:
                    patient_prob["test"]  = softmax_prob_y
                    patient_prob["train"] = softmax_prob_train_y
                    patient_prob["val"]   = softmax_prob_val_y
                    all_prob[patient_id] = patient_prob 
                
        all_patient_cms.append(repeat_patient_cms)
        all_cms.append(cm.detach().clone())
        all_confs.append(repeat_confs) 
    else:
        logger.info(f"Net was not pretrained in base model.")

    all_performance = get_performance_metrics(torch.stack(all_cms).sum(dim=0))

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
        info="Shadow Model for Ensemble Validation Approach on real data"
    )

    logger.info(f"All repeats done. Saving now ...")
    save_results_ens_real(model, mechanism, all_prob, all_performance, config)


def main():
    train_ens_val_shadow_model()
    #train_ens_val_shadow_real_data()

if __name__ == "__main__":
    main()