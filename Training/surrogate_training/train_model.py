import argparse
import os
import numpy as np
import matplotlib
import pandas as pd

matplotlib.use("Agg")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
import horovod.keras as hvd ### importing horovod to use data parallelization in another step

from ST_funcs.clr_callback import *
from ST_funcs.smiles_regress_transformer_funcs import *
from tensorflow.python.client import device_lib
import json
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config",
    type=Path,
    required=True,
    help="config file",
)
args = parser.parse_args()

json_file = args.config#'config_training.json'
hyper_params = ParamsJson(json_file)

'''
Set up horovod if necessary
'''
if hyper_params['general']['use_hvd']==True:
    initialize_hvd()

'''
Create training 
and validation data
'''

x_train, y_train, x_val, y_val = train_val_data(hyper_params)

''' 
Build model architecture 
and set up callbacks
'''
model = ModelArchitecture(hyper_params).call()
model.summary()
train_and_callbacks = TrainingAndCallbacks(hyper_params)

''' Train and save model'''
history = train_and_callbacks.training(
    model,
    x_train,
    y_train,
    (x_val, y_val),
    hyper_params
    )

save_model(model, hyper_params['callbacks']['checkpt_file'], hyper_params['general']['model_out'])
