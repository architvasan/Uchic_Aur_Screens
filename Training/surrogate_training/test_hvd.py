import sys
import argparse
import os
import numpy as np
import pandas as pd
import json
from functools import partial
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
import codecs
from SmilesPE.tokenizer import *
from ST_funcs.smiles_pair_encoders_functions import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
import horovod.keras as hvd ### importing horovod to use data parallelization in another step
from ST_funcs.clr_callback import *
from tensorflow.python.client import device_lib
from itertools import chain, repeat, islice
from mpi4py import MPI

def initialize_hvd():
    hvd.init() 
    print("I am rank %d of %d" %(hvd.rank(), hvd.size()))
    
    #HVD-2: GPU pinning
    gpus = tf.config.experimental.list_physical_devices('XPU')
    print(gpus)
    # Ping GPU to each9 rank
    for gpu in gpus:
    	tf.config.experimental.set_memory_growth(gpu,True)
    if gpus:
    	tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'XPU')

    return 

initialize_hvd()

print(hvd.rank())
