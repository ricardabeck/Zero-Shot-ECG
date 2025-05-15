#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from scipy import linalg as spl
import matplotlib.pyplot as plt

import os
from os.path import join as osj
import sys
import pickle

from progress_bar import print_progress

import torch
from torch import nn


# ## Everything related to loading ECG datasets.

# In[2]:


def load_dict_beats(PATH):
    with open(PATH, "rb") as f:
        return pickle.load(f)
    
def load_data_beats(PATH):
    with open(PATH, "rb") as f:
        return pickle.load(f)
    
def ensure_normalized_and_detrended(beats):
    for key in beats.keys():
        b = beats[key]["beats"]
        if not np.allclose(np.linalg.norm(b, axis=1, ord=2), 1):
            raise AssertionError(f"Beats of patient {key} is not normalized.")
            
        p = np.polyfit(np.arange(b.shape[1]), b.T, deg=1)
        if not np.allclose(p, 0):
            raise AssertionError(f"Beats of patient {key} is not detrended.")


# In[3]:


def load_dataset(patient_id, PATH):
    """ 
    Reads the pickled ECG dataset from the given path for the given patient.
    The file name must be "patient_<patient_id>_dataset.pkl".
    """
    with open(osj(PATH, f"patient_{patient_id}_dataset.pkl"), "rb") as f:
        return pickle.load(f)
    
def load_dictionary(patient_id, PATH):
    """
    Reads the pickled ECG dictionary from the given path for the given patient.
    The file name must be "patient_<patient_id>_dictionary.pkl".
    """
    with open(osj(PATH, f"patient_{patient_id}_dictionary.pkl"), "rb") as f:
        D = pickle.load(f)
        F = spl.null_space(D.T)
        F = spl.orth(F).T
    return D, F

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

def add_dataset(patient_id, dataset, DATASET_PATH):
    """
    Adds another dataset to an already existing one, increasing the number of channels.
    """
    dataset = dataset.copy()
    dataset_other = load_dataset(patient_id, DATASET_PATH)
    dataset_other = dataset_to_tensors(dataset_other)
    
    assert torch.equal(dataset["train_y"], dataset_other["train_y"]), "Training ground truths are different. Possibly shuffled differently."
    assert torch.equal(dataset["val_y"], dataset_other["val_y"]), "Validation ground truths are different. Possibly shuffled differently."
    assert torch.equal(dataset["test_y"], dataset_other["test_y"]), "Test ground truths are different. Possibly shuffled differently."
    
    train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids = dataset.values()
    train_other_X, _, _, val_other_X, _, _, test_other_X, _, _ = dataset_other.values()
    dataset["train_X"] = torch.cat((train_X, train_other_X), dim=1)
    dataset["val_X"] = torch.cat((val_X, val_other_X), dim=1)
    dataset["test_X"] = torch.cat((test_X, test_other_X), dim=1)
    return dataset

def load_N_channel_dataset(patient_id, DEFAULT_PATH, *PATHS):
    """
    Loads the ECG dataset at the given path(s) for the given patient. Each dataset will be added as a new
    channel in the given order.
    """
    default_dataset = load_dataset(patient_id, DEFAULT_PATH)
    default_dataset = dataset_to_tensors(default_dataset)
    for PATH in PATHS:
        default_dataset = add_dataset(patient_id, default_dataset, PATH)
    return default_dataset


# ## Dictionary errors from the annihilator matrix.

# In[4]:


def get_error_one_patient(S, F, y=None, as_energy=False):
    """
    Returns the error, E = S @ F.T
    
    Parameters
    ----------
    S : array_like
        ECG signals of shape (N x K), where K is signal length.
    F : array_like
        Dictionary null-space of shape (M x K), where M is error signal length.
    y : array_like
        Array of labels of shape (N). If provided, errors are returned separately for healthy and arrhythmia.
    as_energy : bool
        If True, error energies are returned instead.
    """
    
    assert F.shape[1] == S.shape[1], f"F and S can't be matrix multiplied. Provide S as a matrix with shape (N x {F.shape[1]})."
    assert y is None or len(np.unique(y)) == 2, f"There must be 2 classes. Found {len(np.unique(y))} classes."
    
    E = S @ F.T
    
    if as_energy:
        E = E.pow(2).sum(dim=1)
    
    if y is not None:
        healthy = np.where(y == 0)[0]
        arrhyth = np.where(y == 1)[0]

        E_healthy = E[healthy]
        E_arrhyth = E[arrhyth]
        
        return E, E_healthy, E_arrhyth
    return E

def get_error_per_patient(S, ids, DICT_PATH, y=None, as_energy=False):
    """
    Returns the error, E = S_i @ F_i.T, where S_i and F_i are the signals and null-space of patient i.
    
    Parameters
    ----------
    S : array_like
        ECG signals of shape (N x K), where K is signal length.
    ids : array_like
        ids[i] is the id of the patient that S[i] belongs to.
    DICT_PATH : str
        Path to the folder that has the dictionary of the users.
    y : array_like
        Array of labels of shape (N). If provided, errors are returned separately for healthy and arrhythmia.
    as_energy : bool
        If True, error energies are returned instead.
    """
    
    _, F = load_dictionary(ids[0], DICT_PATH)
    F = torch.Tensor(F).float()
    E_shape = S.shape[0] if as_energy else [S.shape[0], F.shape[0]]
    Es = torch.empty(E_shape)
    
    for id_ in ids.unique():
        _, F = load_dictionary(id_, DICT_PATH)
        F = torch.Tensor(F).float()
        idx = np.where(ids == id_)[0]
        E = get_error_one_patient(S[ids == id_], F, as_energy=as_energy)
        Es[idx] = E
    
    if y is not None:
        healthy = np.where(y == 0)[0]
        arrhyth = np.where(y == 1)[0]

        E_healthy = Es[healthy]
        E_arrhyth = Es[arrhyth]
        
        return Es, E_healthy, E_arrhyth
    
    return Es


# ## Model used to get the results in our paper.

# In[5]:


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


# ## Probability classes and functions.

# In[6]:


class ExponentialFit:
    def __init__(self):
        pass
    
    def fit(self, X):
        self.beta = X.mean()
        self.fit_perf = np.sum(self.likelihood(X))
    
    def likelihood(self, X):
        return sp.stats.expon(scale=self.beta).pdf(X)


# In[7]:


class GaussianFit:
    def __init__(self):
        pass
    
    def fit(self, X):
        self.mu = X.mean()
        self.std = X.std()
        self.fit_perf = np.sum(self.likelihood(X))
    
    def likelihood(self, X):
        return sp.stats.norm(loc=self.mu, scale=self.std).pdf(X)


# In[8]:


class KDEFit:
    def __init__(self):
        pass
    
    def fit(self, X):
        self.kde = sp.stats.gaussian_kde(X, bw_method=0.05)
        
    def likelihood(self, X):
        return self.kde.pdf(X)


# In[9]:


class BayesianFit:
    def __init__(self):
        pass
        
    def predict(self, X, model1, model2, prior1=1, prior2=1):
        like1 = model1.likelihood(X) * prior1
        like2 = model2.likelihood(X) * prior2
        odd1 = like1 / like2
        return odd1 / (1 + odd1)


# ## Performance metrics.

# In[10]:


def get_performance_metrics(cm):
    """
    Calculates:
        accuracy
        true positive rate (recall, sensitivity)
        specificity (1 - false positive rate)
        positive predictive value (PPV, precision)
        negative predictive value (NPV)
        F1-score
    from the given confusion matrix.
    """
    """cm = np.asarray(cm).copy().astype(np.long)"""
    cm = np.asarray(cm).copy().astype(np.longlong)
    tp, fp, tn, fn = cm[0,0], cm[0,1], cm[1,1], cm[1,0]
    acc = (tp + tn) / (tp + tn + fp + fn)
    rec = tp / (tp + fn)
    spe = tn / (tn + fp)
    pre = tp / (tp + fp)
    npv = tn / (tn + fn)
    f1 = (2 * pre * rec) / (pre + rec)
    metrics = {"acc":acc, "rec":rec, "spe":spe, "pre":pre, "npv":npv, "f1":f1}
    return metrics

def get_confusion_matrix(pred_y, true_y, pos_is_zero=False):
    """
    Calculates the confusion matrix for the given predictions and truth values. 
    
    Set pos_is_zero to True if the positive sample's class index is 0.
    In the case of our ECG work, positive means an abnormal beat, and has a class index of 1.
    """
    pred_y = torch.as_tensor(pred_y, dtype=torch.long)
    true_y = torch.as_tensor(true_y, dtype=torch.long)
    vals = true_y + 2 * pred_y   # 0,0 -> 0    1,0 -> 1    0,1 -> 2    1,1 -> 3
    cm = torch.zeros(4).long()  
    cm += torch.bincount(vals, minlength=4)
    cm = cm.reshape(2, 2)
    
    if not pos_is_zero:
        return cm.flip((0, 1))
    return cm

# @deprecated
def get_confusion_matrix_deprecated(pred_y, true_y, pos_is_zero=False):
    cm = torch.zeros(2, 2).long()
    for py, ty in zip(pred_y, true_y):
        if not pos_is_zero:
            py = 1 - py
            ty = 1 - ty
        cm[py, ty] += 1
    return cm

# @deprecated
def get_cm_generating_data_deprecated(cm):
    """ 
    Previously needed for WandB. 
    Creates pseudo-data that would generate the given confusion matrix.
    """
    cm = np.asarray(cm).copy().astype(np.long)
    tp, fp, tn, fn = cm[0,0], cm[0,1], cm[1,1], cm[1,0]
    pred_y = torch.cat((torch.ones(tp), torch.zeros(tn), torch.ones(fp), torch.zeros(fn))).long()
    test_y = torch.cat((torch.ones(tp), torch.zeros(tn), torch.zeros(fp), torch.ones(fn))).long()
    return pred_y, test_y


# ## Dictionary and Sparse Key generation.

# **Solves the Lasso problem:**
# $$ \underset{\vec{x}}{\arg\min} \dfrac{1}{2}\|\mathbf{A}\vec{x} - \vec{b}\|_2^2 + \lambda\|\vec{x}\|_1 $$
# 
# **using [Alternating Direction Method of Multipliers](https://statweb.stanford.edu/~candes/teaching/math301/Lectures/Consensus.pdf). In our case, this corresponds to:**
# 
# $$ \underset{\vec{h}}{\arg\min} \|\mathbf{D}\vec{h} - \vec{x}\|_2^2 + \lambda\|\vec{h}\|_1 $$
# 
# **where $\mathbf{D}$ is the dictionary matrix, $\vec{h}$ is the keys, and $\vec{x}$ is the signal represented using $\mathbf{D}\vec{h}$. We can also represent an entire dataset of signals as $\mathbf{D}\mathbf{H} = \mathbf{X}$.**

# In[11]:


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
                    print_progress(k + 1, max_it, add_newline=True)
                    print(f"Stopping early. Abs error diff: {np.abs(E[k - 1] - E[k]):.2e}, Threshold: {E[k - 1] * delta_err:.2e}")
                k = k + 1
                return D, X, E[:k]
            
        k = k + 1
        if printing:
            print_progress(k, max_it)
        
    if calc_err:
        return D, X, E
    return D, X

def lasso_solver_ADMM(A, b, max_it=100):
    """
    Minimizes the lasso formulation |Ax - b|_2 + |x|_1 using ADMM.
    """
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


# ### Domain Adaptation from one patient to another. <br> Problem: Given two patients, modify the beats of one patient to improve its beat representation on another patient's dictionary.
# ### $$ 
# \begin{equation}
# \begin{aligned}
# \hat{Q}, \hat{C}_j &= \underset{Q,\,C_j}{\arg\min} \, J \\
# J &=  \|QX_j - D_iC_j\|_2^2 + \|C_j\|_1 + \gamma\|X_j - QX_j\|_2^2 \\
# \dfrac{\partial J}{\partial Q} &= 2\left( QX_j - D_iC_j \right)X_j^T - 2\gamma\left( X_j - Q_jX_j \right)X_j^T \\
#   &= Q_jX_jX_j^T - D_iC_jX_j^T - \gamma X_jX_j^T + \gamma Q_jX_jX_j^T = 0 \\
# Q &= \left( D_iC_jX_j^T + \gamma X_jX_j^T \right) \left( X_jX_j^T + \gamma X_jX_j^T \right)^{-1}
# \end{aligned}
# \end{equation}
# $$ <br> where $Q$ is the matrix that transforms $X_j$ to another domain so that it is better represented with $D_i$.

# In[12]:


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

