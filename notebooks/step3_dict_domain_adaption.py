"""
To run in background

    conda activate torch_dpl
    cd notebooks/
    nohup python -m step3_dict_domain_adaption > logs/step3.log 2>&1
    nohup python -m step3_dict_domain_adaption > logs/step3_mp.log 2>&1
    nohup python -m step3_dict_domain_adaption > logs/step3_mp2.log 2>&1
    nohup python -m step3_dict_domain_adaption > logs/step3_true_dp.log 2>&1

Runtime:
    ** dictionaries:
        ~ 25 minutes per mechanism 
        ~ 15 seconds per epsilon
    ** domain adaption:
        ~ 9 hours per mechanism 
        ~ 5 minutes per epsilon
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
# import copy
# import random
# import import_ipynb
# import os
# import sys
# import json
from bisect import bisect
from collections import defaultdict 

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

import concurrent.futures

# Paper Libraries for functions
from ecg_utilities import *
from progress_bar import print_progress

# Pytorch libraries
import torch.nn.functional as Func
from pytorch_sklearn import NeuralNetwork
from pytorch_sklearn.callbacks import WeightCheckpoint, Verbose, LossPlot, EarlyStopping, Callback, CallbackInfo
from pytorch_sklearn.utils.func_utils import to_safe_tensor


def read_5min_beats(s, m): # read dict beats
    with open(osj("..", f"dp_data_{s}", "dataset_beats", f"{m}_5min_normal_beats.pkl"), "rb") as f:
        return pickle.load(f)

def read_25min_beats(s, m): # read data beats
    with open(osj("..", f"dp_data_{s}", "dataset_beats", f"{m}_25min_beats.pkl"), "rb") as f:
        return pickle.load(f)

def ensure_normalized_and_detrended(beats):
    for key in beats.keys():
        b = beats[key]["beats"]
        if not np.allclose(np.linalg.norm(b, axis=1, ord=2), 1):
            raise AssertionError(f"Beats of patient {key} is not normalized.")
            
        p = np.polyfit(np.arange(b.shape[1]), b.T, deg=1)
        if not np.allclose(p, 0):
            raise AssertionError(f"Beats of patient {key} is not detrended.")
        
def generate_dictionary(S, num_atoms=20, delta_err=None, max_it=100, max_admm=100, calc_err=False, seed=0, printing=False):
    """
    Generate a dictionary that represents the signals given in S ∈ (N x F) with sparse keys, minimizing the Lasso loss.
    D ∈ (F x num_atoms).
    X ∈ (num_atoms x N).
    
    Parameters
    ----------
    S : array_like
        Signals to be represented. S ∈ (N x F).
    num_atoms : int
        Number of dictionary elements.
    delta_err : float, default=None
        Stopping criteria for change in error. Stops if the decrease in error is less than (current error * delta_err): 
            Stops if: err(t-1) - err(t) < err(t-1) * delta_err
        Note: calc_err must be true. If calc_err is False, max_it is used. If max_it is reached, we stop regardless.
    """
    assert delta_err is None or calc_err, "Early stopping is not possible without error calculations."  # delta_err implies calc_err
        
    np.random.seed(seed)
    S = S.T  # S ∈ (F x N)
    D = np.random.randn(S.shape[0], num_atoms)
    D = D / np.linalg.norm(D, axis=0, ord=2)
    k = 0
    if calc_err:
        E = np.zeros(max_it)
    
    while k < max_it:
        X = lasso_solver_ADMM(D, S, max_it=max_admm)
        D = S @ np.linalg.pinv(X)  # DX = S  ->  DXX+ = SX+  ->  D = SX+            (Y @ S') \ (S @ S')
        D = D / np.linalg.norm(D, axis=0, ord=2)
        
        if calc_err:
            err = np.linalg.norm((D @ X) - S, ord=2, axis=None)
            E[k] = err
            if k > 1 and delta_err is not None and np.abs(E[k - 1] - E[k]) < E[k - 1] * delta_err:
                if printing:
                    print(f"Stopping early. Abs error diff: {np.abs(E[k - 1] - E[k]):.2e}, Threshold: {E[k - 1] * delta_err:.2e}")
                k = k + 1
                return D, X, E[:k]
            
        k = k + 1
        
    if calc_err:
        return D, X, E
    return D, X

def lasso_solver_ADMM(A, b, max_it=100):
    x = np.zeros((A.shape[1], b.shape[1]))
    z = np.zeros_like(x)
    y = np.zeros_like(x)
    AtA = A.T @ A
    I = np.eye(AtA.shape[0])
    Atb = A.T @ b
    tau_ = 0.08  # one over tau
    lambda_tau = 0.01 / tau_  # lambda * tau
    k = 0
    
    while k < max_it:
        x = np.linalg.solve(AtA + tau_ * I, Atb + tau_ * (z - y))
        z = soft_threshold(x + y, lambda_tau)
        y = y + tau_ * (x - z)
        k = k + 1
        
    return x

def soft_threshold(x, lambda_):
    """
    Implements:
        x - lambda    if x > lambda
        x + lambda    if x < -lambda
        0             otherwise (x in [-lambda, lambda])
    """
    return np.maximum(0, x - lambda_) - np.maximum(0, -lambda_ - x)


def generate_dictionary_pp(patient_ids, dict_beats):

    Ds = {}

    for patient_id in patient_ids:
        dict_beat = dict_beats[patient_id]["beats"]
        D, X, E = generate_dictionary(dict_beat, num_atoms=20, delta_err=None, max_it=20, max_admm=100, calc_err=True, seed=patient_id, printing=True)
        
        # sort D in decreasing order, based on the l1-norm of X's rows.
        sortidx = np.argsort(np.abs(X).sum(axis=1))[::-1]
        D = D[:, sortidx]
        X = X[sortidx, :]
        
        F = spl.null_space(D.T)
        F = spl.orth(F).T
        
        Ds[patient_id] = D
        
    return Ds

def save_dictionary(data, m):
    with open(osj("..", "dp_data_single", "dictionaries", m + "_dictionary.pkl"), "wb") as f:
        pickle.dump(data, f)

class DomainAdapter():
    def __init__(self):
        self.adapted = False
    
    def adapt(self, Di, Sj, gamma: int, max_it: int=100, max_admm: int=100, plot=False):
        Q = np.eye(Di.shape[0])
        Sj = Sj.T
        
        # Save for speedup
        SjSjT = Sj @ Sj.T
        gSjSjT = gamma * SjSjT

        if plot:
            plt.figure()
        for i in range(max_it):
            QSj = Q @ Sj
            if plot:
                plt.plot(QSj[:, 0], label=f"iter={i}")
            QSj = QSj / np.linalg.norm(QSj, axis=0, ord=2)
            Cj = lasso_solver_ADMM(Di, QSj, max_it=max_admm)
            Q = (Di @ Cj @ Sj.T + gSjSjT) @ np.linalg.pinv(SjSjT + gSjSjT)
            
        if plot:
            plt.legend()
            plt.show()

        self.Q = Q
        self.adapted = True
        return Q
    
    def adapt_gd(self, Di, Sj, gamma: int, lr: float=0.01, max_epochs: int=100, max_admm: int=100, plot=False):
        Q = np.eye(Di.shape[0])
        Sj = Sj.T

        # Save for speedup
        SjSjT = Sj @ Sj.T
        gSjSjT = gamma * SjSjT

        if plot:
            plt.figure()
        for i in range(max_epochs):
            QSj = Q @ Sj
            if plot and i % 10 == 0:
                plt.plot(QSj[:, 0], label=f"iter={i}")
            QSj = QSj / np.linalg.norm(QSj, axis=0, ord=2)
            Cj = lasso_solver_ADMM(Di, QSj, max_it=max_admm)
            
            grad_Q = Q @ SjSjT - Di @ Cj @ Sj.T - gSjSjT + gamma * Q @ SjSjT
            Q = Q - lr * grad_Q
        
        if plot:
            plt.legend()
            plt.show()
        
        self.Q = Q
        self.adapted = True
        return Q
    
    def test(self, Di, Si_D, Si_T, yi, Sj_D, Sj_T, yj, Q, max_it: int=100):
        assert self.adapted, "Call adapt first."
        
        # Fit of i's dictionary beats.
        Ci_D = lasso_solver_ADMM(Di, Si_D.T, max_it=max_it)
        Ei_D = np.linalg.norm((Di @ Ci_D) - Si_D.T, ord=2, axis=0).mean()
        
        # Fit of i's train/test beats.
        Ci_T = lasso_solver_ADMM(Di, Si_T.T, max_it=max_it)
        Ei_T = np.linalg.norm((Di @ Ci_T) - Si_T.T, ord=2, axis=0)
        Ei_T_healthy = Ei_T[yi == "N"].mean()
        Ei_T_arrhyth = Ei_T[yi != "N"].mean()
        
        # Fit of j's dictionary beats on i's dictionary.
        Cj_D = lasso_solver_ADMM(Di, Sj_D.T, max_it=max_it)
        Ej_D = np.linalg.norm((Di @ Cj_D) - Sj_D.T, ord=2, axis=0).mean()
        
        # Fit of j's train/test beats on i's dictionary.
        Cj_T = lasso_solver_ADMM(Di, Sj_T.T, max_it=max_it)
        Ej_T = np.linalg.norm((Di @ Cj_T) - Sj_T.T, ord=2, axis=0)
        Ej_T_healthy = Ej_T[yj == "N"].mean()
        Ej_T_arrhyth = Ej_T[yj != "N"].mean()
        
        # Fit of j's dictionary beats on i's dictionary after domain adaptation.
        QSj_D = Q @ Sj_D.T
        QSj_D = QSj_D / np.linalg.norm(QSj_D, axis=0, ord=2)
        
        DA_Cj_D = lasso_solver_ADMM(Di, QSj_D, max_it=max_it)
        DA_Ej_D = np.linalg.norm((Di @ DA_Cj_D) - QSj_D, ord=2, axis=0).mean()
        
        # Fit of j's train/test beats on i's dictionary after domain adaptation.
        QSj_T = Q @ Sj_T.T
        QSj_T = QSj_T / np.linalg.norm(QSj_T, axis=0, ord=2)
        
        DA_Cj_T = lasso_solver_ADMM(Di, QSj_T, max_it=max_it)
        DA_Ej_T = np.linalg.norm((Di @ DA_Cj_T) - QSj_T, ord=2, axis=0)
        DA_Ej_T_healthy = DA_Ej_T[yj == "N"].mean()
        DA_Ej_T_arrhyth = DA_Ej_T[yj != "N"].mean()
        
        d = {
            "Ei_D": Ei_D, "Ei_T_healthy": Ei_T_healthy, "Ei_T_arrhyth": Ei_T_arrhyth,
            "Ej_D": Ej_D, "Ej_T_healthy": Ej_T_healthy, "Ej_T_arrhyth": Ej_T_arrhyth,
            "DA_Ej_D": DA_Ej_D, "DA_Ej_T_healthy": DA_Ej_T_healthy, "DA_Ej_T_arrhyth": DA_Ej_T_arrhyth
        }
        
        return d
    
def get_patient_adaptation_matrix(Ds, dict_5min, gamma, lr, max_epochs):
    DA = DomainAdapter()
    Q = DA.adapt_gd(Ds, dict_5min["beats"], gamma=gamma, lr=lr, max_epochs=max_epochs)
    return Q


def get_patient_healthy_and_arrhythmia(data_25min, patient_id, healthy_leq_arrhyth=True):
    """
    Returns healthy and arrhythmia beats of patient, where number of healthy returned <= number of arrhythmia returned.
    """
    data_beat = data_25min[patient_id]["beats"]
    data_class = data_25min[patient_id]["class"]
    idx_healthy = np.where(data_class == "N")[0]
    idx_arrhyth = np.where(data_class != "N")[0]
    num_healthy = len(idx_healthy)
    num_arrhyth = len(idx_arrhyth)
    
    if healthy_leq_arrhyth and num_healthy > num_arrhyth:
        num_healthy = num_arrhyth  # take only as much as arrhythmias, maybe try to take all healthies?
    
    healthies = data_beat[idx_healthy[:num_healthy], ...]
    arrhythmias = data_beat[idx_arrhyth[:num_arrhyth], ...]
    return healthies, arrhythmias

def get_patient_train_data(Ds, dict_5min, data_25min, patient_id, valid_patients):
    """
    For patient p, prepare p's train data by getting other patients' healthy and arrhythmia beats + p's dictionary beats. 
    Healthy beats must be <= arrhythmia beats.
    """
    train_X = []
    train_y = []
    train_ids = []
    for i, other_id in enumerate(valid_patients):
        other_healthy, other_arrhyth = get_patient_healthy_and_arrhythmia(data_25min, other_id)
        
        # Ds, beats, i, j, gamma, lr, max_epochs
        Q = get_patient_adaptation_matrix(Ds[patient_id], dict_5min[other_id], gamma=0.2, lr=0.002, max_epochs=25)
        other_healthy = (Q @ other_healthy.T)
        other_healthy = other_healthy / np.linalg.norm(other_healthy, axis=0, ord=2)
        other_healthy = other_healthy.T
        
        other_arrhyth = (Q @ other_arrhyth.T)
        other_arrhyth = other_arrhyth / np.linalg.norm(other_arrhyth, axis=0, ord=2)
        other_arrhyth = other_arrhyth.T
        
        train_X.append(other_healthy)
        train_y.append(np.zeros(len(other_healthy)))
        train_X.append(other_arrhyth)
        train_y.append(np.ones(len(other_arrhyth)))
        train_ids.append(np.ones(len(other_healthy) + len(other_arrhyth)) * other_id)
    dict_beat = dict_5min[patient_id]["beats"]
    train_X.append(dict_beat)
    train_y.append(np.zeros(len(dict_beat)))
    train_ids.append(np.ones(len(dict_beat)) * patient_id)
    
    return np.concatenate(train_X, axis=0), np.concatenate(train_y, axis=0), np.concatenate(train_ids, axis=0)

def get_patient_test_data(data_25min, patient_id):
    """
    For patient p, prepare p's test data from p's 25 minute beats (i.e. beats that are not dictionary beats).
    """
    data_beat = data_25min[patient_id]["beats"]
    data_class = data_25min[patient_id]["class"]
    idx_healthy = np.where(data_class == "N")[0]
    idx_arrhyth = np.where(data_class != "N")[0]
    
    test_X = [data_beat[idx_healthy], data_beat[idx_arrhyth]]
    test_y = [np.zeros(len(idx_healthy)), np.ones(len(idx_arrhyth))]
    test_ids = np.ones(len(idx_healthy) + len(idx_arrhyth)) * patient_id
    
    return np.concatenate(test_X, axis=0), np.concatenate(test_y, axis=0), test_ids

def shuffle(X, y, ids, seed=None):
    """
    Shuffle X, y and ids with the same indices, and optionally set a seed.
    """
    if seed is not None:
        np.random.seed(seed)
    shuffle_idx = np.random.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx], ids[shuffle_idx]

def train_validation_split(train_X, train_y, train_ids, ratio):
    """
    Train/Validation split by the given ratio, where ratio is size_train / size_all. Keeps the ratio of healthy and arrhythmia beats the same in train and
    validation sets.
    """
    idx_healthy = np.where(train_y == 0)[0]
    idx_arrhyth = np.where(train_y == 1)[0]
    
    num_healthy = len(idx_healthy)
    num_arrhyth = len(idx_arrhyth)
    
    num_train_healthy = int(num_healthy * ratio)
    num_train_arrhyth = int(num_arrhyth * ratio)
    
    num_val_healthy = num_healthy - num_train_healthy
    num_val_arrhyth = num_arrhyth - num_train_arrhyth
    
    val_X = np.concatenate((train_X[idx_healthy[0:num_val_healthy]], train_X[idx_arrhyth[0:num_val_arrhyth]]))
    val_y = np.concatenate((train_y[idx_healthy[0:num_val_healthy]], train_y[idx_arrhyth[0:num_val_arrhyth]]))
    val_ids = np.concatenate((train_ids[idx_healthy[0:num_val_healthy]], train_ids[idx_arrhyth[0:num_val_arrhyth]]))
    
    train_X = np.concatenate((train_X[idx_healthy[num_val_healthy:]], train_X[idx_arrhyth[num_val_arrhyth:]]))
    train_y = np.concatenate((train_y[idx_healthy[num_val_healthy:]], train_y[idx_arrhyth[num_val_arrhyth:]]))
    train_ids = np.concatenate((train_ids[idx_healthy[num_val_healthy:]], train_ids[idx_arrhyth[num_val_arrhyth:]])) 
    
    return train_X, train_y, train_ids, val_X, val_y, val_ids

def generate_patient_datasets(Ds, dict_5min, data_25min, patient_ids):
    """
    Combines all of the functions above to generate datasets for each patient.
    """
    all_patients_dataset = {patient_id:{"train_X":[], "train_y":[], "train_ids":[], "val_X":[], "val_y":[], "val_ids":[], "test_X":[], "test_y":[], "test_ids":[]} for patient_id in patient_ids}

    for i, patient_id in enumerate(patient_ids):

        train_X, train_y, train_ids = get_patient_train_data(Ds, dict_5min, data_25min, patient_id, patient_ids)
        train_X, train_y, train_ids = shuffle(train_X, train_y, train_ids, seed=patient_id)
        
        test_X, test_y, test_ids = get_patient_test_data(data_25min, patient_id)
        test_X, test_y, test_ids = shuffle(test_X, test_y, test_ids, seed=None)
        
        train_X, train_y, train_ids, val_X, val_y, val_ids = train_validation_split(train_X, train_y, train_ids, ratio=0.8)
        train_X, train_y, train_ids = shuffle(train_X, train_y, train_ids, seed=None)
        val_X, val_y, val_ids = shuffle(val_X, val_y, val_ids, seed=None)
        
        all_patients_dataset[patient_id]["train_X"]   = train_X
        all_patients_dataset[patient_id]["train_y"]   = train_y
        all_patients_dataset[patient_id]["train_ids"] = train_ids
        all_patients_dataset[patient_id]["val_X"]     = val_X
        all_patients_dataset[patient_id]["val_y"]     = val_y
        all_patients_dataset[patient_id]["val_ids"]   = val_ids
        all_patients_dataset[patient_id]["test_X"]    = test_X
        all_patients_dataset[patient_id]["test_y"]    = test_y
        all_patients_dataset[patient_id]["test_ids"]  = test_ids

    return all_patients_dataset

def save_domain_adapted(data, setup, m):
    with open(osj("..", f"dp_data_{setup}", "dataset_training", m + "_domain_adapted.pkl"), "wb") as f:
        pickle.dump(data, f)

def load_domain_adapted(s,m): 
    with open(osj("..", f"dp_data_{s}", "dataset_training", f"{m}_domain_adapted.pkl"), "rb") as f:
        return pickle.load(f)

def load_dict_data(m):
    with open(osj("..", f"dp_data_single", "dictionaries", f"{m}_dictionary.pkl"), "rb") as f:
        return pickle.load(f)

def generate_dictionaries():

    valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)
    # p_method = ["laplace", "bounded_n", "gaussian_a", "laplace_truedp"]
    p_method = ["laplace_truedp"]
    
    dictionaries_all = {}

    for mechanism in p_method:

        dict_5min  = read_5min_beats("single", mechanism)

        if mechanism == "bounded_n":
            hp_epsilon_values = [0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                            0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                            1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91,
                            2.01, 2.11, 2.21, 2.31, 2.41, 2.51, 2.61, 2.71, 2.81, 2.91,
                            3.01, 3.11, 3.21, 3.31, 3.41, 3.51, 3.61, 3.71, 3.81, 3.91,
                            4.01, 4.11, 4.21, 4.31, 4.41, 4.51, 4.61, 4.71, 4.81, 4.91,
                            5.01, 5.11, 5.21, 5.31, 5.41, 5.51, 5.61, 5.71, 5.81, 5.91,
                            6.01, 6.11, 6.21, 6.31, 6.41, 6.51, 6.61, 6.71, 6.81, 6.91,
                            7.01, 7.11, 7.21, 7.31, 7.41, 7.51, 7.61, 7.71, 7.81, 7.91,
                            8.01, 8.11, 8.21, 8.31, 8.41, 8.51, 8.61, 8.71, 8.81, 8.91,
                            9.01, 9.11, 9.21, 9.31, 9.41, 9.51, 9.61, 9.71, 9.81, 9.91, 10]
        elif mechanism == "laplace_truedp":
            hp_epsilon_values = [0.00001, 0.0001, 0.001, 
                                0.011, 0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91]
        else:
            hp_epsilon_values = [0.00001, 0.0001, 0.001, 
                                0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                            0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                            1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91,
                            2.01, 2.11, 2.21, 2.31, 2.41, 2.51, 2.61, 2.71, 2.81, 2.91,
                            3.01, 3.11, 3.21, 3.31, 3.41, 3.51, 3.61, 3.71, 3.81, 3.91,
                            4.01, 4.11, 4.21, 4.31, 4.41, 4.51, 4.61, 4.71, 4.81, 4.91,
                            5.01, 5.11, 5.21, 5.31, 5.41, 5.51, 5.61, 5.71, 5.81, 5.91,
                            6.01, 6.11, 6.21, 6.31, 6.41, 6.51, 6.61, 6.71, 6.81, 6.91,
                            7.01, 7.11, 7.21, 7.31, 7.41, 7.51, 7.61, 7.71, 7.81, 7.91,
                            8.01, 8.11, 8.21, 8.31, 8.41, 8.51, 8.61, 8.71, 8.81, 8.91,
                            9.01, 9.11, 9.21, 9.31, 9.41, 9.51, 9.61, 9.71, 9.81, 9.91, 10]
        
        if os.path.exists(osj("..", "dp_data_single", "dictionaries", f"{mechanism}_dictionary.pkl")):
            dictionaries_all = load_dict_data(mechanism)

        skipped = True
        ########  EPSILON  DICT ########
        for epsilon in hp_epsilon_values:
            if len(hp_epsilon_values) == len(dictionaries_all.keys()):
                logger.info(f"All {epsilon} are complete -- ")
                continue
            elif epsilon in dictionaries_all.keys():
                logger.info(f"Skipping epsilon {epsilon} -- ")
                continue
            else:  
                skipped = False

                logger.info(f"Creating dictionary for epsilon {epsilon} ...")

                ensure_normalized_and_detrended(dict_5min[epsilon])
                dictionaries_all[epsilon] = generate_dictionary_pp(valid_patients, dict_5min[epsilon])

        if skipped == False:
            logger.info(f"Savind dictionary for mechanism {mechanism} ...")  
            save_dictionary(dictionaries_all, mechanism)


def generate_domain_adaption():

    valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)
    # p_method = ["laplace", "bounded_n", "gaussian_a", "laplace_truedp"]
    p_method = ["laplace_truedp"]
    p_beats = ["single", "trio"] 
    for mechanism in p_method:

        if mechanism == "bounded_n":
            hp_epsilon_values = [0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91,
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
                                0.011, 0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91]
        else:
            hp_epsilon_values = [0.00001, 0.0001, 0.001, 
                                0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91,
                                2.01, 2.51, 2.91,
                                3.01, 3.51, 3.91,
                                4.01, 4.51, 4.91,
                                5.01, 5.51, 5.91,
                                6.01, 6.51, 6.91,
                                7.01, 7.51, 7.91,
                                8.01, 8.51, 8.91,
                                9.01, 9.51, 9.91, 10]
            
        for setup in p_beats:
            logger.info(f"Starting with {setup} beats for {mechanism} ...")
            dict_5min  = read_5min_beats(setup, mechanism)
            data_25min = read_25min_beats(setup, mechanism)
            dictionaries_all = load_dict_data(mechanism)
            domain_adapted_all = {}

            if os.path.exists(osj("..", f"dp_data_{setup}", "dataset_training", f"{mechanism}_domain_adapted.pkl")):
                try:
                    domain_adapted_all = load_domain_adapted(setup, mechanism)
                except EOFError:
                    logger.info("No domain adapted dataset found.")
            
            ########  EPSILON DOMAIN  ########
            for epsilon in hp_epsilon_values:

                if epsilon in domain_adapted_all.keys():
                    logger.info(f"Skipping epsilon {epsilon} -- ")
                else:
                    logger.info(f"Starting domain adaption for epsilon {epsilon} ...")
                    ensure_normalized_and_detrended(dict_5min[epsilon])
                    ensure_normalized_and_detrended(data_25min[epsilon])

                    domain_adapted_e = generate_patient_datasets(Ds=dictionaries_all[epsilon], dict_5min=dict_5min[epsilon], data_25min=data_25min[epsilon], patient_ids=valid_patients) # Takes around 5 minutes to generate all datasets. PER EPSILON
                    domain_adapted_all[epsilon] = domain_adapted_e

                    save_domain_adapted(domain_adapted_all, setup, mechanism)
                    logger.info(f"Saved domain adaption for {setup} beats until epsilon {epsilon}.")


def main():

    generate_dictionaries()
    generate_domain_adaption()

if __name__ == "__main__":
    main()




# def start_multiprocess():
    
#     method = {}
#     num_processes = 3
#     method[0] = "laplace"
#     method[1] = "bounded_n"
#     method[2] = "gaussian_a"


#     with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
#         futures = {executor.submit(generate_domain_adaption, method[i], i): i for i in range(num_processes)}
    
#     for future in concurrent.futures.as_completed(futures):
#         process_id = futures[future]
#         try:
#             future.result()
#             logger.info(f"Finished processing in process {process_id}.")

#         except Exception as e:
#             logger.error(f"Error processing in process {process_id}: {e}")

# def dictionary_domain_adaption():

#     valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)

#     ########  MECHANISM  ########
#     for mechanism in p_method:
#         logger.info(f" Setup for {mechanism} ...")

#         dictionaries_all = {} # = Ds

#         ########  SETUP  ########
#         for setup in p_beats:
#             logger.info(f" Loading data for {setup} beats ...")

#             dict_5min  = read_5min_beats(setup, mechanism)
#             data_25min = read_25min_beats(setup, mechanism)
#             domain_adapted_all = {}
            
#             ########  EPSILON  DICT ########
#             for epsilon in hp_epsilon_values:
#                 ensure_normalized_and_detrended(dict_5min[epsilon])
#                 ensure_normalized_and_detrended(data_25min[epsilon])

#                 if setup == "single":
#                     logger.info(f"Generate dictionaries for epsilon {epsilon} ...")
#                     dictionaries_all[epsilon] = generate_dictionary_pp(valid_patients, dict_5min[epsilon])
            
#             if setup == "single":
#                 save_dictionary(dictionaries_all, mechanism)

#             ########  EPSILON  DOMAIN ########
#             for epsilon in hp_epsilon_values:
#                 logger.info(f"Starting domain adaption for epsilon {epsilon} ...")
#                 domain_adapted_e = generate_patient_datasets(Ds=dictionaries_all[epsilon], dict_5min=dict_5min[epsilon], data_25min=data_25min[epsilon], patient_ids=valid_patients) # Takes around 5 minutes to generate all datasets. PER EPSILON
#                 domain_adapted_all[epsilon] = domain_adapted_e

#             logger.info(f"Saving files for {setup} beats ...")
#             save_domain_adapted(domain_adapted_all, setup, mechanism)
