"""
To run:
    conda activate torch_dpl
    cd notebooks/
    nohup python -m step1_DPapplication > logs/step1_laplace.log 2>&1
    nohup python -m step1_DPapplication > logs/step1_laplace_trueDP.log 2>&1
    nohup python -m step1_DPapplication > logs/step1_bounded_n.log 2>&1
    nohup python -m step1_DPapplication > logs/step1_gaussian_a.log 2>&1

Runtime:
    6 seconds per patient
    5 minutes per epsilon
    
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import concurrent.futures

# Differential privacy libraries
from diffprivlib import mechanisms
from diffprivlib import models
from diffprivlib import tools
from diffprivlib.utils import check_random_state
from diffprivlib.mechanisms import Laplace, LaplaceBoundedNoise, GaussianAnalytic
from diffprivlib.mechanisms import DPMechanism

# Paper Libraries for functions
from ecg_utilities import *
from progress_bar import print_progress

# Pytorch libraries
import torch.nn.functional as Func
from pytorch_sklearn import NeuralNetwork
from pytorch_sklearn.callbacks import WeightCheckpoint, Verbose, LossPlot, EarlyStopping, Callback, CallbackInfo
from pytorch_sklearn.utils.func_utils import to_safe_tensor

ROOT = osj("..", "physionet.org/files/mitdb/1.0.0")
RECORDS = osj(ROOT, "RECORDS")


## MIT Data Processing Functions
def get_patient_ids():
    patient_ids = pd.read_csv(RECORDS, delimiter="\n", header=None).to_numpy().reshape(-1)
    return patient_ids

def get_ecg_signals(patient_ids):
    """
    The MIT-BIH data was generated with 2 leads. Per patient this function reads the ecg data per lead.
    """
    lead0 = {}
    lead1 = {}
    for id_ in patient_ids:
        signals, _ = wfdb.io.rdsamp(osj(ROOT, str(id_)))
        lead0[id_] = signals[:, 0]
        lead1[id_] = signals[:, 1]
    return lead0, lead1

def get_ecg_info(patient_ids):
    """
    The MIT-BIH data additionally contains ecg info, providing additional information, such as the age, sex, gender, and comments.
    """
    info = {}
    for id_ in patient_ids:
        _, info_ = wfdb.io.rdsamp(osj(ROOT, str(id_)))
        info[id_] = info_["comments"][0]
    return info

def get_paced_patients(patient_ids):
    """
    The MIT-BIH records have 4 patients, that used a pacemaker and hence should be excluded later on.
    """
    paced = []
    for id_ in patient_ids:
        annotation = wfdb.rdann(osj(ROOT, str(id_)), extension='atr')
        labels = np.unique(annotation.symbol)
        if ("/" in labels):
            paced.append(id_)
    return np.array(paced)

def get_all_beat_labels(patient_ids):
    """
    Getting the unique set of labels that are present in the MIT data.
    """
    all_labels = []
    for id_ in patient_ids:
        annotation = wfdb.rdann(osj(ROOT, str(id_)), extension='atr')
        labels = np.unique(annotation.symbol)
        all_labels.extend(labels)
    return np.unique(all_labels)

def get_rpeaks_and_labels(patient_ids):
    """
    Getting the ids of the r-peaks and the corresponding annotated label.
    """
    rpeaks = {}
    labels = {}
    for id_ in patient_ids:
        annotation = wfdb.rdann(osj(ROOT, str(id_)), extension='atr')
        rpeaks[id_] = annotation.sample
        labels[id_] = np.array(annotation.symbol)
    return rpeaks, labels

def get_normal_beat_labels():
    """
    The MIT-BIH labels that are classified as healthy/normal. Check wfdb.Annotation documentation for description of labels.
    N: {N, L, R, e, j}. 
    """
    return np.array(["N", "L", "R", "e", "j"])

def get_abnormal_beat_labels():
    """
    The MIT-BIH labels that are classified as arrhythmia/abnormal. Check wfdb.Annotation documentation for description of labels.
    S: {S, A, J, a} - V: {V, E} - F: {F} - Q: {Q}
    """
    return np.array(["S", "A", "J", "a", "V", "E", "F", "Q"])


## DP specific functions
def get_global_sensitivity(valid_patients, lead0, labels):

    all_count = {patient: {"normal": 0, "abnormal": 0, "min": 0.0, "max": 0.0, "mean": 0.0} for patient in valid_patients}

    normal_labels = get_normal_beat_labels()
    abnormal_labels = get_abnormal_beat_labels()

    # get values per patient
    for patient in valid_patients:
        p_min_value = 0
        p_max_value = 0
        p_mean_value = 0

        count_normal = len(np.where(np.isin(labels[patient], normal_labels))[0])
        count_abnormal = len(np.where(np.isin(labels[patient], abnormal_labels))[0])
        p_mean_value = np.mean(lead0[patient])
        p_min_value = np.min(lead0[patient])
        p_max_value = np.max(lead0[patient])

        all_count[patient]['normal'] = count_normal
        all_count[patient]['abnormal'] = count_abnormal
        all_count[patient]['min'] = p_min_value
        all_count[patient]['max'] = p_max_value
        all_count[patient]['mean'] = p_mean_value

    # aggregate values while iteratively leaving one patient out
    all_count_agg = {patient: {"g_ratio": 0.0, "g_min": 0.0, "g_max": 0.0, "g_mean": 0.0} for patient in valid_patients}
    for patient_leavout in valid_patients:

        all_count_copy = copy.deepcopy(all_count)
        del all_count_copy[patient_leavout] 
        values = all_count_copy.values()

        normal = sum(patient["normal"] for patient in values)
        abnormal = sum(patient["abnormal"] for patient in values)
        ratio = abnormal / normal

        all_count_agg[patient_leavout]["g_ratio"] = ratio
        all_count_agg[patient_leavout]["g_min"]   = min(patient["min"] for patient in values)
        all_count_agg[patient_leavout]["g_max"]   = max(patient["max"] for patient in values)
        sum_mean                                  = sum(patient["mean"] for patient in values)
        all_count_agg[patient_leavout]["g_mean"]  = sum_mean / len(values)


    agg_values = all_count_agg.values()
    ratio_difference = max(patient["g_ratio"] for patient in agg_values)     - min(patient["g_ratio"] for patient in agg_values)
    min_difference   = abs(min(patient["g_min"] for patient in agg_values))  - abs(max(patient["g_min"] for patient in agg_values)) # abs(min) is bigger than abs(max)
    max_difference   = max(patient["g_max"] for patient in agg_values)       - min(patient["g_max"] for patient in agg_values)
    mean_difference  = abs(min(patient["g_mean"] for patient in agg_values)) - abs(max(patient["g_mean"] for patient in agg_values)) # abs(min) is bigger than abs(max)

    return ratio_difference

def set_dp_mechanism(m, e, d, s): 
    seed = random.seed(42)
    if m == 'laplace':
        dp_mechanism = Laplace(epsilon=e, delta=d, sensitivity=s, random_state=seed)
    elif m == 'laplace_truedp':
        dp_mechanism = Laplace(epsilon=e, delta=0.0, sensitivity=s, random_state=seed)
    elif m == 'bounded_n':
        dp_mechanism = LaplaceBoundedNoise(epsilon=e, delta=d, sensitivity=s, random_state=seed) # Delta must be > 0 and in (0, 0.5).
    elif m == "gaussian_a":
        dp_mechanism = GaussianAnalytic(epsilon=e, delta=d, sensitivity=s, random_state=seed)

    return dp_mechanism

def run_diffpriv(method, valid_patients, lead0, epsilon, sensitivity):
    ecgs = copy.deepcopy(lead0)
    i = 0
    random.seed(42) # ensures only that the same ecg values are changed, but the results will differ
    # mechanism = set_dp_mechanism(method, epsilon, 0.49, sensitivity) # original delta value
    mechanism = set_dp_mechanism(method, epsilon, 0.0, sensitivity) # added true DP

    ########  PATIENT  ########
    for patient in valid_patients: 
        logger.info(f"Starting with patient {patient} ...")
        i += 1
        signal_count = 0 

        ########  SIGNAL  ########
        for signal in ecgs[patient]:
            dp_signal = mechanism.randomise(signal)
            ecgs[patient][signal_count] = dp_signal   
            signal_count += 1

    return ecgs

def read_dp_signals(m):
    with open(osj("..", "dp_signals", m + ".pkl"), "rb") as f:
        return pickle.load(f)

def save_dp_signals(dict_signals_dp, m):
    with open(osj("..", "dp_signals", m + ".pkl"), "wb") as f:
        pickle.dump(dict_signals_dp, f)

# def save_dp_signals_process(dict_signals_dp, m, pid):
#     with open(osj("..", "dp_signals", f"{m}_{pid}.pkl"), "wb") as f:
#         pickle.dump(dict_signals_dp, f)
    
# def read_dp_signals(m):
#     with open(osj("..", "dp_signals", m + ".pkl"), "rb") as f:
#         return pickle.load(f)

# def read_dp_signals_process(m, pid):
#     with open(osj("..", "dp_signals", f"{m}_{pid}.pkl"), "rb") as f:
#         return pickle.load(f)
    


def read_and_diffpriv():

    p_method = ["laplace", "bounded_n", "gaussian_a", "laplace_truedp"]
    
    hp_epsilon_values = [0.00001, 0.0001, 0.001, 
                        0.01, 0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                            0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                        1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91,
                        2.01, 2.11, 2.21, 2.31, 2.41, 2.51, 2.61, 2.71, 2.81, 2.91,
                        3.01, 3.11, 3.21, 3.31, 3.41, 3.51, 3.61, 3.71, 3.81, 3.91,
                        4.01, 4.11, 4.21, 4.31, 4.41, 4.51, 4.61, 4.71, 4.81, 4.91,
                        5.01, 5.11, 5.21, 5.31, 5.41, 5.51, 5.61, 5.71, 5.81, 5.91,
                        6.01, 6.11, 6.21, 6.31, 6.41, 6.51, 6.61, 6.71, 6.81, 6.91,
                        7.01, 7.11, 7.21, 7.31, 7.41, 7.51, 7.61, 7.71, 7.81, 7.91,
                        8.01, 8.11, 8.21, 8.31, 8.41, 8.51, 8.61, 8.71, 8.81, 8.91,
                        9.01, 9.11, 9.21, 9.31, 9.41, 9.51, 9.61, 9.71, 9.81, 9.91, 10]
    hp_epsilon_values_bounded = [0.01, 0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                            0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                        1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91,
                        2.01, 2.11, 2.21, 2.31, 2.41, 2.51, 2.61, 2.71, 2.81, 2.91,
                        3.01, 3.11, 3.21, 3.31, 3.41, 3.51, 3.61, 3.71, 3.81, 3.91,
                        4.01, 4.11, 4.21, 4.31, 4.41, 4.51, 4.61, 4.71, 4.81, 4.91,
                        5.01, 5.11, 5.21, 5.31, 5.41, 5.51, 5.61, 5.71, 5.81, 5.91,
                        6.01, 6.11, 6.21, 6.31, 6.41, 6.51, 6.61, 6.71, 6.81, 6.91,
                        7.01, 7.11, 7.21, 7.31, 7.41, 7.51, 7.61, 7.71, 7.81, 7.91,
                        8.01, 8.11, 8.21, 8.31, 8.41, 8.51, 8.61, 8.71, 8.81, 8.91,
                        9.01, 9.11, 9.21, 9.31, 9.41, 9.51, 9.61, 9.71, 9.81, 9.91, 10]
    hp_epsilon_values_truedp = [0.00001, 0.0001, 0.001, 
                                0.011, 0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091,
                                0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91,
                                1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91]
        
    # 1 Read MIT data
    logger.info(f" Reading the MIT data")
    patient_ids = get_patient_ids()
    lead0, lead1 = get_ecg_signals(patient_ids) # total 650.000 values in lead0 per patient
    rpeaks, labels = get_rpeaks_and_labels(patient_ids)

    # 2 Apply Differential Privacy
    sensitivity = get_global_sensitivity(patient_ids, lead0, labels)

    for mechanism in p_method:
        logger.info(f" Adding differential privacy with {mechanism} for all patients per epsilon.")

        results = read_dp_signals(mechanism)
        # results = {}

        if mechanism == "bounded_n":
            hp_epsilon_values = hp_epsilon_values_bounded
        elif mechanism == "laplace_truedp":
            hp_epsilon_values = hp_epsilon_values_truedp
    
        for epsilon in hp_epsilon_values:

            try:
                results[epsilon]
                logger.info(f"Data for epsilon {epsilon} already exists. Skipping...")
                continue
            except KeyError:
                results[epsilon] = None
                logger.info(f"New epsilon {epsilon} added")
            
                logger.info(f"Calculating data for epsilon {epsilon} ...")
                results[epsilon] = run_diffpriv(mechanism, patient_ids, lead0, epsilon, sensitivity)        
        
        # 3 save per mechanism
        save_dp_signals(results, mechanism)
        logger.info(f"DP data saved.")


def main():
    read_and_diffpriv()

if __name__ == "__main__":
    main()


# Multiprocessing version

# def process_epsilon_batch(epsilon_batch, mechanism, valid_patients, lead0, sensitivity, process_id):
#     """Funktion, die eine Gruppe von Epsilon-Werten verarbeitet und speichert."""
#     logger.info(f"Starting processing for process {process_id}")

#     results = {}
    
#     for epsilon in epsilon_batch:

#         logger.info(f"Calculating data for epsilon {epsilon} in process {process_id} ...")
#         results[epsilon] = run_diffpriv(mechanism, valid_patients, lead0, epsilon, sensitivity)

#     logger.info(f"Returning results of process {process_id}.")
#     return results

# def read_and_diffpriv():
    
#     # 1 Read MIT data
#     logger.info(f" Reading the MIT data")
#     patient_ids = get_patient_ids()
#     lead0, lead1 = get_ecg_signals(patient_ids) # total 650.000 values in lead0 per patient
#     rpeaks, labels = get_rpeaks_and_labels(patient_ids)

#     # 2 Apply Differential Privacy
#     ########  MECHANISM  ########
#     sensitivity = get_global_sensitivity(patient_ids, lead0, labels)

#     for mechanism in p_method:
#         logger.info(f" Adding differential privacy with {mechanism} for all patients per epsilon.")

#         # # read_pickle
#         # if os.path.exists(osj("..", "dp_signals", mechanism + ".pkl")):
#         #     dict_signals_dp = read_dp_signals(mechanism)

#         # split remaining epsilons per process
#         num_processes = 12
#         batch_size = len(hp_epsilon_values) // num_processes
#         epsilon_batches = [hp_epsilon_values[i * batch_size: (i + 1) * batch_size] for i in range(num_processes)]
#         # Falls Werte übrig bleiben (wegen Rundung), hänge sie an das letzte Batch an
#         if len(hp_epsilon_values) % num_processes != 0:
#             epsilon_batches[-1].extend(hp_epsilon_values[num_processes * batch_size:])

#         merged_results = {}  # Gemeinsames Dictionary für alle Prozesse

#         # Parallelisierung mit begrenzter Anzahl an Prozessen
#         with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
#             futures = {executor.submit(process_epsilon_batch, epsilon_batches[i], mechanism, patient_ids, lead0, sensitivity, i): i 
#                        for i in range(num_processes)}
            
#             for future in concurrent.futures.as_completed(futures):
#                 process_id = futures[future]
#                 try:
#                     process_results = future.result()
#                     logger.info(f"Finished processing batch in process {process_id}.")
#                     merged_results.update(process_results)

#                 except Exception as e:
#                     logger.error(f"Error processing batch in process {process_id}: {e}")

#         save_dp_signals_process(merged_results, mechanism)

