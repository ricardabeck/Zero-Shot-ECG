"""
To run:

    conda activate torch_dpl
    cd notebooks/
    nohup python -m step6b_train_DA_shadow > logs/step6_DA_shadow_dp.log 2>&1
    nohup python -m step6b_train_DA_shadow > logs/step6_DA_shadow_real.log 2>&1

Runtime:
    ~ 13 minutes per epsilon (instead of 1 hour with the original model)

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


def load_train_test_data(m, e):
    with open(osj("..", "dp_models", "train_test_data", m, f"{e}_data.pkl"), "rb") as f:
        data = pickle.load(f)
    return data 

def load_real_train_test_data():
    with open(osj("..", "dp_models", "train_test_data", "real_data.pkl"), "rb") as f:
        data = pickle.load(f)
    return data 

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

def save_new_results(model, mechanism, e, prob, p, c, n): 
    # save performance
    with open(osj("..", "dp_models", model, mechanism, f"{e}_performance.pkl"), "wb") as f:
        pickle.dump(p, f) 
    
    # save probabilities
    with open(osj("..", "dp_models", model, mechanism, f"{e}_probs.pkl"), "wb") as f:
        pickle.dump(prob, f) 
            
    # save config
    with open(osj("..", "dp_models", model, mechanism, f"{e}_config.pkl"), "wb") as f: 
        pickle.dump(c, f) 
        
    # save model
    with open(osj("..", "dp_models", model, mechanism, f"{e}_nets.pkl"), "wb") as f:
        pickle.dump(n, f) 

def save_updated_results(model, mechanism, e, prob, p, c):
    # save performance
    with open(osj("..", "dp_models", model, mechanism, f"{e}_performance.pkl"), "wb") as f:
        pickle.dump(p, f) 
    
    # save probabilities
    with open(osj("..", "dp_models", model, mechanism, f"{e}_probs.pkl"), "wb") as f:
        pickle.dump(prob, f) 
            
    # save config
    with open(osj("..", "dp_models", model, mechanism, f"{e}_config.pkl"), "wb") as f: 
        pickle.dump(c, f) 

def save_results_real(model, mechanism, prob, p, c, n):
    # save performance
    with open(osj("..", "dp_models", model, mechanism, f"performance.pkl"), "wb") as f:
        pickle.dump(p, f) 
    
    # save probabilities
    with open(osj("..", "dp_models", model, mechanism, f"probs.pkl"), "wb") as f:
        pickle.dump(prob, f) 
            
    # save config
    with open(osj("..", "dp_models", model, mechanism, f"config.pkl"), "wb") as f: 
        pickle.dump(c, f) 
        
    # save model
    with open(osj("..", "dp_models", model, mechanism, f"nets.pkl"), "wb") as f:
        pickle.dump(n, f) 

def load_net(mechanism, e):
    with open(osj("..", "dp_models", "DA_shadow", mechanism, f"{e}_nets.pkl"), "rb") as f:
        net_e = pickle.load(f)
    return net_e


###### Shadow Training ######
def train_shadow_model():

    p_method = ["laplace", "bounded_n", "gaussian_a", "laplace_truedp"]

    attack_setup = pd.read_pickle("../files/attack_setup.pkl") # dataframe with columns: Model, Method, Epsilon, Metric, Value   
    # no additional model filtering, since DA needs to train all epsilons, those for DA and for Ens_val

    model = "DA_shadow" 
    max_epochs = [100]
    batch_sizes = [1024]
    repeats = 10

    valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)
  
    ########  MECHANISM  ########
    for mechanism in p_method:
        logger.info(f"Setup for {mechanism} ...")
        hp_epsilon_values = list(set(attack_setup[attack_setup["Method"] == mechanism]["Epsilon"].tolist())) # deduplicate epsilons from both models

        ########  EPSILON  ########
        for epsilon in hp_epsilon_values:

            # ------------ Work with the pretrained model ------------
            if os.path.exists(osj("..", "dp_models", model, mechanism, f"{epsilon}_nets.pkl")):
                logger.info(f"Pretrained model available for epsilon {epsilon}, no training needed.")
                
                continue

            # ------------ Train a new model ------------
            else:
                net_e = None
                
                # Check if data is prepared for the shadow model
                if os.path.exists(osj("..", "dp_models", "train_test_data", mechanism, f"{epsilon}_data.pkl")):
                    logger.info(f"Getting data for epsilon {epsilon} ...")

                    train_test_data = load_train_test_data(mechanism, epsilon)
                    epsilon_patient_cms = []
                    epsilon_cms = []
                    epsilon_net = {}
                    epsilon_performance = {}
                    epsilon_prob = dict.fromkeys(valid_patients)
                    
                    ########  REPEATS  ########
                    for repeat in range(repeats):
                        logger.info(f"Repeat {repeat+1}/10 for epsilon {epsilon}")

                        repeat_patient_cms = {}
                        repeat_net = {}

                        cm = torch.zeros(2, 2)
                        
                        for i, patient_id in enumerate(valid_patients):
                            logger.info(f"Training for patient {patient_id} ...")

                            patient_net = {}
                            patient_prob = dict.fromkeys("test", "train")

                            train_X   = train_test_data[patient_id]["Val_x"]
                            train_y   = train_test_data[patient_id]["Val_y"]
                            test_X    = train_test_data[patient_id]["Test_x"]
                            test_y    = train_test_data[patient_id]["Test_y"]

                            # Neural Network setup
                            model_base = get_base_model(in_channels=train_X.shape[1])
                            model_base = model_base.to("cuda")
                            crit = nn.CrossEntropyLoss()
                            optim = torch.optim.AdamW(params=model_base.parameters())

                            net = NeuralNetwork(model_base, optim, crit)
                            weight_checkpoint_train_loss = WeightCheckpoint(tracked="train_loss", mode="min")
                            early_stopping = EarlyStopping(tracked="train_loss", mode="min", patience=15)

                            # Neural Network training
                            net.fit(
                                train_X=train_X,
                                train_y=train_y,
                                validate=False,
                                max_epochs=max_epochs[0],
                                batch_size=batch_sizes[0],
                                use_cuda=True,
                                fits_gpu=True,
                                callbacks=[weight_checkpoint_train_loss, early_stopping],
                            )
                        
                            net.load_weights(weight_checkpoint_train_loss)
                            pred_y = net.predict(test_X, batch_sizes[0], use_cuda=True, fits_gpu=True, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
                            prob_y = net.predict_proba(test_X).cpu()
                            prob_train_y = net.predict_proba(train_X).cpu()
                            softmax_prob_y = Func.softmax(prob_y, dim=1).max(dim=1).values
                            softmax_prob_train_y = Func.softmax(prob_train_y, dim=1).max(dim=1).values

                            if repeat == 9:
                                patient_prob["test"]  = softmax_prob_y
                                patient_prob["train"] = softmax_prob_train_y
                                epsilon_prob[patient_id] = patient_prob

                            patient_net = NeuralNetwork.return_class(net) # save best model for this patient, in this repeat
                            repeat_net[patient_id] = patient_net

                            cur_cm = get_confusion_matrix(pred_y, test_y, pos_is_zero=False)
                            repeat_patient_cms[patient_id] = cur_cm # patient_cm
                            cm += cur_cm 
                            
                        epsilon_net[repeat] = repeat_net
                        epsilon_patient_cms.append(repeat_patient_cms)
                        epsilon_cms.append(cm.detach().clone())  # Konvertiert in eine Python-Liste

                else:
                    
                    logger.info(f"Train-test data was not prepared for epsilon {epsilon}.")
                
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
                checkpoint_on=weight_checkpoint_train_loss.tracked,
                dataset="default+trio",
                info="2-channel run, domain adapted, consulting with default dictionary, and trying all thresholds, saves weights"
            )

            if net_e == None: # save everything for new model
                logger.info(f"All repeats for epsilon {epsilon} done. Saving prob, performance, config, net.")
                save_new_results(model, mechanism, epsilon, epsilon_prob, epsilon_performance, config, epsilon_net)
            else:
                logger.info(f"All repeats for epsilon {epsilon} done. Saving prob, performance, config.")
                save_updated_results(model, mechanism, epsilon, epsilon_prob, epsilon_performance, config)

def shadow_model_real_data():

    model = "DA_shadow" 
    mechanism = "no_dp"
    max_epochs = [100]
    batch_sizes = [1024]
    repeats = 10
    valid_patients = pd.read_csv(osj("..", "files", "valid_patients.csv"), header=None).to_numpy().reshape(-1) 

    all_patient_cms = []
    all_cms = []
    all_nets = {}
    all_performance = {}
    all_prob = dict.fromkeys(valid_patients)

    train_test_data = load_real_train_test_data()

    ########  REPEATS  ########
    for repeat in range(repeats):
        logger.info(f"Repeat {repeat+1}/10")

        repeat_patient_cms = {}
        repeat_net = {}
        cm = torch.zeros(2, 2)
        
        for i, patient_id in enumerate(valid_patients):
            logger.info(f"Training for patient {patient_id} ...")

            patient_net = {}
            patient_prob = dict.fromkeys("test", "train")

            train_X   = train_test_data[patient_id]["Val_x"]
            train_y   = train_test_data[patient_id]["Val_y"]
            test_X    = train_test_data[patient_id]["Test_x"]
            test_y    = train_test_data[patient_id]["Test_y"]

            # Neural Network setup
            model_base = get_base_model(in_channels=train_X.shape[1])
            crit = nn.CrossEntropyLoss()
            optim = torch.optim.AdamW(params=model_base.parameters())

            net = NeuralNetwork(model_base, optim, crit)
            weight_checkpoint_train_loss = WeightCheckpoint(tracked="train_loss", mode="min")
            early_stopping = EarlyStopping(tracked="train_loss", mode="min", patience=15)

            # Neural Network training
            net.fit(
                train_X=train_X,
                train_y=train_y,
                validate=False,
                max_epochs=max_epochs[0],
                batch_size=batch_sizes[0],
                use_cuda=True,
                fits_gpu=True,
                callbacks=[weight_checkpoint_train_loss, early_stopping],
            )
        
            net.load_weights(weight_checkpoint_train_loss)
            pred_y = net.predict(test_X, batch_sizes[0], use_cuda=True, fits_gpu=True, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
            
            prob_y = net.predict_proba(test_X).cpu()
            prob_train_y = net.predict_proba(train_X).cpu()
            softmax_prob_y = Func.softmax(prob_y, dim=1).max(dim=1).values
            softmax_prob_train_y = Func.softmax(prob_train_y, dim=1).max(dim=1).values

            if repeat == 9:
                patient_prob["test"]  = softmax_prob_y
                patient_prob["train"] = softmax_prob_train_y
                all_prob[patient_id] = patient_prob

            patient_net = NeuralNetwork.return_class(net) # save best model for this patient, in this repeat
            repeat_net[patient_id] = patient_net

            cur_cm = get_confusion_matrix(pred_y, test_y, pos_is_zero=False)
            repeat_patient_cms[patient_id] = cur_cm # patient_cm
            cm += cur_cm 
            
        all_nets[repeat] = repeat_net
        all_patient_cms.append(repeat_patient_cms)
        all_cms.append(cm.detach().clone())  # Konvertiert in eine Python-Liste

    all_cms = np.stack(all_cms).astype(int)
    all_performance = get_performance_metrics(all_cms.sum(axis=0))

    # after all repeats (per epsilon)
    config = dict(
        learning_rate=0.001,
        max_epochs=max_epochs[0],
        batch_size=batch_sizes[0],
        optimizer=optim.__class__.__name__,
        loss=crit.__class__.__name__,
        early_stopping="true",
        checkpoint_on=weight_checkpoint_train_loss.tracked,
        dataset="default+trio",
        info="2-channel run, domain adapted, consulting with default dictionary, and trying all thresholds, saves weights"
    )

    logger.info(f"All repeats done. Saving now ...")
    save_results_real(model, mechanism, all_prob, all_performance, config, all_nets)


def main():
    train_shadow_model()
    #shadow_model_real_data()

if __name__ == "__main__":
    main()