"""
To run:

    conda activate torch_dpl
    cd notebooks/
    nohup python -m step2_single_trio_beats > logs/step2_beats_dp_all.log 2>&1
    nohup python -m step2_single_trio_beats > logs/step2_rerun.log 2>&1
    nohup python -m step2_single_trio_beats > logs/step2_beats_dp_more.log 2>&1

Runtime:
    ~ 50 minutes per mechanism 
    ~ 15 seconds per epsilon

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
import concurrent.futures


# Paper Libraries for functions
from ecg_utilities import *
from progress_bar import print_progress

# Pytorch libraries
import torch.nn.functional as Func
from pytorch_sklearn import NeuralNetwork
from pytorch_sklearn.callbacks import WeightCheckpoint, Verbose, LossPlot, EarlyStopping, Callback, CallbackInfo
from pytorch_sklearn.utils.func_utils import to_safe_tensor

ROOT = osj("..", "physionet.org/files/mitdb/1.0.0")
p_method = ["laplace", "bounded_n", "gaussian_a", "laplace_truedp"]
# p_method = ["laplace_truedp"]
p_beats = ["single", "trio"]
p_data = ['5min_normal_beats', '25min_beats'] # to distinguish between 5 minutes for training and the remainng 25 minutes for testing


def get_valid_patients():
    valid_patient_ids = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)
    return valid_patient_ids

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

def find_fivemin_index(patient_ids, rpeaks):
    #secs = 20
    #samps = secs * fs
    fs = 360
    fivemin = fs * 60 * 5 # 108.000
    fivemin_index = []
    for patient_id in patient_ids:
        idx = bisect(rpeaks[patient_id], fivemin) - 1
        fivemin_index.append(idx)
    return fivemin_index

def read_dp_signals(m):
    with open(osj("..", "dp_signals", m + ".pkl"), "rb") as f:
        return pickle.load(f)

def get_beat_class(label):
    """
    A mapping from labels to classes, based on the rules described in get_normal_beat_labels() and get_abnormal_beat_labels().
    """
    if label in ["N", "L", "R", "e", "j"]:
        return "N"
    elif label in ["S", "A", "J", "a"]:
        return "S"
    elif label in ["V", "E"]:
        return "V"
    elif label == "F" or label == "Q":
        return label
    return None

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

def get_beats(patient_ids, signals, rpeaks, labels, beat_trio=False, centered=False, lr_offset=0.1, matlab=False):
    """
    For each patient:
    Converts its ECG signal to an array of valid beats, where each rpeak with a valid label is converted to a beat of length 128 by resampling (Fourier-Domain).
    Converts its labels to an array of valid labels, and a valid label is defined in the functions get_normal_beat_labels() and get_abnormal_beat_labels().
    Converts its valid labels to an array of classes, where each valid label is one of 5 classes, (N, S, V, F, Q).
    
    Parameters
    ----------
    beat_trio: bool, default=False
        If True, generate beats as trios.
        
    centered: bool, default=False
        Whether the generated beats have their peaks centered.
        
    lr_offset: float, default=0.1, range=[0, 1]
        A beat is extracted by finding the beats before and after it, and then offsetting by some samples. This parameter controls how many samples are
        offsetted. If the lower beat is L, and the current beat is C, then we offset by `lr_offset * abs(L - C)` samples.
        
    matlab: bool, default=False
        If True, dictionary keys become strings to be able to save the dictionary as a .mat file.
    """
    
    beat_length = 128
    get_key_name = lambda patient_id: f"patient_{patient_id}" if matlab else patient_id

    beat_data = {get_key_name(patient_id):{"beats":[], "class":[], "label":[]} for patient_id in patient_ids}
    
    for j, patient_id in enumerate(patient_ids):
        key_name = get_key_name(patient_id)
        
        # Filter out rpeaks that do not correspond to a valid label.
        valid_labels = np.concatenate((get_normal_beat_labels(), get_abnormal_beat_labels()))
        valid_idx = np.where(np.isin(labels[patient_id], valid_labels))[0]
        valid_rpeaks = rpeaks[patient_id][valid_idx]
        valid_labels = labels[patient_id][valid_idx]
        
        for i in range(1, len(valid_rpeaks) - 1):
            lpeak = valid_rpeaks[i - 1]
            cpeak = valid_rpeaks[i]
            upeak = valid_rpeaks[i + 1]
    
            if beat_trio:
                lpeak = int(lpeak - (lr_offset * abs(cpeak - lpeak)))
                upeak = int(upeak + (lr_offset * abs(cpeak - upeak)))
            else:
                lpeak = int(lpeak + (lr_offset * abs(cpeak - lpeak)))
                upeak = int(upeak - (lr_offset * abs(cpeak - upeak)))
            
            if centered:
                ldiff = abs(lpeak - cpeak)
                udiff = abs(upeak - cpeak)
                diff = min(ldiff, udiff)
                
                # Take same number of samples from the center.
                beat = signals[patient_id][cpeak - diff:cpeak + diff + 1]
            else:
                beat = signals[patient_id][lpeak:upeak]
            
            # Resampling in the frequency domain instead of in the time domain (resample_poly)
            # beat = sp.signal.resample_poly(beat, beat_length, len(beat))
            beat = sp.signal.resample(beat, beat_length)
    
            # detrend the beat and normalize it.
            beat = sps.detrend(beat)
            beat = beat / np.linalg.norm(beat, ord=2)
        
            label = valid_labels[i]
        
            beat_data[key_name]["beats"].append(beat)
            beat_data[key_name]["class"].append(get_beat_class(label))
            beat_data[key_name]["label"].append(label)
        beat_data[key_name]["beats"] = np.stack(beat_data[key_name]["beats"])
        beat_data[key_name]["class"] = np.stack(beat_data[key_name]["class"])
        beat_data[key_name]["label"] = np.stack(beat_data[key_name]["label"])
        
        #print_progress(j + 1, len(patient_ids), opt=[f"{patient_id}"])
    return beat_data

def create_fivemin_remaining_beats(patient_ids, fivemin_index, beat_data):
    fivemin_beat_data = {patient_id:{"beats":[], "class":[], "label":[]} for patient_id in patient_ids}
    remaining_beat_data = {patient_id:{"beats":[], "class":[], "label":[]} for patient_id in patient_ids}

    for i, patient_id in enumerate(patient_ids):

        # normal beats under 5 minutes
        normal_idx = np.where(beat_data[patient_id]["class"][0:fivemin_index[i]] == "N")[0]
        # all other beats
        other_idx = np.setdiff1d(np.arange(0, len(beat_data[patient_id]["class"])), normal_idx)
        
        assert len(normal_idx) + len(other_idx) == len(beat_data[patient_id]["class"]), "Some beats are not taken into account!"

        # no errors, continue to store the beats, classes and labels.
        fivemin_beat_data[patient_id]["beats"] = beat_data[patient_id]["beats"][normal_idx, :]
        fivemin_beat_data[patient_id]["class"] = beat_data[patient_id]["class"][normal_idx]
        fivemin_beat_data[patient_id]["label"] = beat_data[patient_id]["label"][normal_idx]

        assert np.count_nonzero(fivemin_beat_data[patient_id]["class"] != "N") == 0, "Abnormal beat misplaced!"
        
        remaining_beat_data[patient_id]["beats"] = beat_data[patient_id]["beats"][other_idx, :]
        remaining_beat_data[patient_id]["class"] = beat_data[patient_id]["class"][other_idx]
        remaining_beat_data[patient_id]["label"] = beat_data[patient_id]["label"][other_idx]

    return fivemin_beat_data, remaining_beat_data

def save_5min_beats(data,setup, m):
    with open(osj("..", f"dp_data_{setup}", "dataset_beats", m + "_5min_normal_beats.pkl"), "wb") as f:
        pickle.dump(data, f)

def save_25min_beats(data, setup, m):
    with open(osj("..", f"dp_data_{setup}", "dataset_beats", m + "_25min_beats.pkl"), "wb") as f:
        pickle.dump(data, f)

def save_30min_beats(data, setup, m):
    with open(osj("..", f"dp_data_{setup}", "dataset_beats", m + "_30min_beats.pkl"), "wb") as f:
        pickle.dump(data, f)

def read_5min_beats(setup, m):
    with open(osj("..", f"dp_data_{setup}", "dataset_beats", m + "_5min_normal_beats.pkl"), "rb") as f:
        data = pickle.load(f)
    return data

def read_25min_beats(setup, m):
    with open(osj("..", f"dp_data_{setup}", "dataset_beats", m + "_25min_beats.pkl"), "rb") as f:
        data = pickle.load(f)
    return data

def read_30min_beats(setup, m):
    with open(osj("..", f"dp_data_{setup}", "dataset_beats", m + "_30min_beats.pkl"), "rb") as f:
        data = pickle.load(f)
    return data

def create_beats_split_minutes():

    valid_patients = get_valid_patients()
    rpeaks, labels = get_rpeaks_and_labels(valid_patients)
    fivemin_index = find_fivemin_index(valid_patients, rpeaks)

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

    ########  MECHANISM  ########
    for mechanism in p_method:
        logger.info(f"Loading the dataset for {mechanism}.")
        dict_signals = read_dp_signals(mechanism)

        ########  SETUP  ########
        for setup in p_beats:
            logger.info(f"Starting with {setup} beats dataset")
            if setup == "single":
                trio = False
            elif setup == "trio":
                trio = True

            if mechanism == "bounded_n":
                hp_epsilon_values = hp_epsilon_values_bounded
            elif mechanism == "laplace_truedp":
                hp_epsilon_values = hp_epsilon_values_truedp
                

            if os.path.exists(osj("..", f"dp_data_{setup}", "dataset_beats", f"{mechanism}_30min_beats.pkl")):
                beat_data           = read_30min_beats(setup, mechanism)
                fivemin_beat_data   = read_5min_beats(setup, mechanism)
                remaining_beat_data = read_25min_beats(setup, mechanism)
            else:
                beat_data           = {}
                fivemin_beat_data   = {}
                remaining_beat_data = {}

            skipped  = False
            ########  EPSILON  ########
            for epsilon in hp_epsilon_values:

                try:
                    beat_data[epsilon]
                    logger.info(f"Data for epsilon {epsilon} already exists. Skipping...")
                    skipped  = True
                    continue
                except KeyError:
                    beat_data[epsilon] = None
                    logger.info(f"New epsilon {epsilon} added, satrting to process...")
                    skipped  = False

                    beat_data[epsilon] = get_beats(valid_patients, dict_signals[epsilon], rpeaks, labels, beat_trio = trio, centered=False, lr_offset=0.1)
                    fivemin_beat_data[epsilon], remaining_beat_data[epsilon] = create_fivemin_remaining_beats(valid_patients, fivemin_index, beat_data[epsilon])
            
            if skipped == False:
                logger.info(f"Saving beats 5, 25, 30 min for {setup} beats with {mechanism}")

                save_5min_beats(fivemin_beat_data, setup, mechanism)
                save_25min_beats(remaining_beat_data, setup, mechanism)
                save_30min_beats(beat_data, setup, mechanism)



def main():
    create_beats_split_minutes()

if __name__ == "__main__":
    main()





##------------------ Code for parallelization ------------------##

# def process_epsilon_batch(epsilon_batch, valid_patients, dict_signals, rpeaks, labels, trio, fivemin_index, i):
#     logger.info(f"Starting processing for process {i}")

#     ########  EPSILON  ########
#     for epsilon in epsilon_batch:
#         logger.info(f" Starting with epsilon: {epsilon}")

#         beat_data = {}
#         fivemin_beat_data = {}
#         remaining_beat_data = {}


#         beat_data[epsilon] = get_beats(valid_patients, dict_signals[epsilon], rpeaks, labels, beat_trio = trio, centered=False, lr_offset=0.1)
#         fivemin_beat_data[epsilon], remaining_beat_data[epsilon] = create_fivemin_remaining_beats(valid_patients, fivemin_index, beat_data[epsilon])

#     return fivemin_beat_data, remaining_beat_data


# def create_beats_split_minutes():

#     valid_patients = get_valid_patients()
#     rpeaks, labels = get_rpeaks_and_labels(valid_patients)
#     fivemin_index = find_fivemin_index(valid_patients, rpeaks)

#     ########  MECHANISM  ########
#     for mechanism in p_method:
#         logger.info(f" Loading the dataset for {mechanism}.")
#         dict_signals = read_dp_signals(mechanism)

#         ########  SETUP  ########
#         for setup in p_beats:
#             logger.info(f" Starting with {setup} beats dataset")
#             if setup == "single":
#                 save_path = osj(BEATS_SAVE_PATH, "dp_data_single")
#                 trio = False
#             elif setup == "trio":
#                 save_path = osj(BEATS_SAVE_PATH, "dp_data_trio")
#                 trio = True

#             # Start Multi-Processing            
#             num_processes = 10
#             batch_size = len(hp_epsilon_values) // num_processes
#             epsilon_batches = [hp_epsilon_values[i * batch_size: (i + 1) * batch_size] for i in range(num_processes)]
#             if len(hp_epsilon_values) % num_processes != 0:
#                 epsilon_batches[-1].extend(hp_epsilon_values[num_processes * batch_size:])

#             merged_fivemin_beat_data = {}
#             merged_remaining_beat_data = {}

#             # Parallelisierung
#             with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
#                 futures = {executor.submit(process_epsilon_batch, epsilon_batches[i], valid_patients, dict_signals, rpeaks, labels, trio, fivemin_index, i): i 
#                         for i in range(num_processes)}
                
#                 for future in concurrent.futures.as_completed(futures):
#                     process_id = futures[future]
#                     try:
#                         five, remaining = future.result()
#                         logger.info(f"Finished processing batch in process {process_id}.")

#                         merged_fivemin_beat_data.update(five)  
#                         merged_remaining_beat_data.update(remaining) 

#                     except Exception as e:
#                         logger.error(f"Error processing batch in process {process_id}: {e}")

#             merged_fivemin_beat_data   = dict(sorted(merged_fivemin_beat_data.items()))
#             merged_remaining_beat_data = dict(sorted(merged_remaining_beat_data.items()))
                
#             save_5min_beats(save_path, merged_fivemin_beat_data, mechanism)
#             save_25min_beats(save_path, merged_remaining_beat_data, mechanism)