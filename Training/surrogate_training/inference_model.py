import argparse
import os
import numpy as np
import matplotlib
import pandas as pd
from mpi4py import MPI
import csv
from collections import OrderedDict

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

from tensorflow.keras.preprocessing import sequence, text
from clr_callback import *
from tensorflow.python.client import device_lib
import json
from ST_funcs.smiles_pair_encoders_functions import *
import time
from ST_funcs.smiles_regress_transformer_funcs import *

json_file = 'config_inference.json'
hyper_params = ParamsJson(json_file)

'''
Load Model + setup mpi
'''

model = tf.keras.models.load_model(hyper_params['model']['path'])
model.summary()

comm, size, rank = initialize_mpi()

'''
Organize data files + setup tokenizer
'''
split_files, split_dirs = large_scale_split(hyper_params, size, rank)

if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
    vocab_file = hyper_params['tokenization']['tokenizer']['vocab_file']
    spe_file = hyper_params['tokenization']['tokenizer']['spe_file']
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

''' 
Iterate over files
'''

BATCH = hyper_params['general']['batch_size']
cutoff = hyper_params['general']['cutoff']

'''
Inference running on each file
'''

for fil, dirs in zip(split_files, split_dirs):

    Data_smiles_inf, x_inference = large_inference_data_gen(hyper_params, tokenizer, dirs, fil, rank)

    Output = model.predict(x_inference, batch_size = BATCH)

    '''
    Combine SMILES and predicted docking score.
    Sort the data based on the docking score, 
    remove data below cutoff score.
    write data to file in output directory
    '''
    SMILES_DS = np.vstack((Data_smiles_inf, np.array(Output).flatten())).T 
    SMILES_DS = sorted(SMILES_DS, key=lambda x: x[1], reverse=True)

    filtered_data = list(OrderedDict((item[0], item) for item in SMILES_DS if item[1] >= cutoff).values())
    filename = f'output/{dirs}/{os.path.splitext(fil)[0]}.{rank%4}.dat'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['smiles', 'score'])
        writer.writerows(filtered_data)

    del (Data_smiles_inf)
    del(Output)
    del(x_inference)
    del(SMILES_DS)
    del(filtered_data)


'''
Sorting all files
parallel merge sort
'''
if True:
    Sorted_data = pd.DataFrame(columns = ['smiles', 'score'])
    
    for fil, dirs in zip(split_files, split_dirs):
        filename = f'output/{dirs}/{os.path.splitext(fil)[0]}.{rank%4}.dat'
        df = pd.read_csv(filename)
        Sorted_data = pd.concat([Sorted_data, df])
    Sorted_data = Sorted_data.to_numpy()
    Sorted_data = sorted(Sorted_data, key=lambda x: x[1], reverse=True)
    
    Sorted_data = comm.gather(Sorted_data, root=0)
    
    if rank==0:
        print(len(Sorted_data))
        data_to_write = Sorted_data[0]
        for r in range(1,len(Sorted_data)):
            data_to_write.extend(Sorted_data[r])
        data_to_write = sorted(data_to_write, key=lambda x: x[1], reverse=True)
        data_to_write = list(OrderedDict((item[0], item) for item in data_to_write).values())
    
        filename = f'All.sorted.dat'
    
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['smiles', 'score'])
            writer.writerows(data_to_write[0:10000000])
    
    
