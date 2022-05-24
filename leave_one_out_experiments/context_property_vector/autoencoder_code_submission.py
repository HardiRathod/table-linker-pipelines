###########################################################
### Importing Libraries ###
print("Importing Libraries .......")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import glob
import os
from ast import literal_eval
import scipy.sparse as sp
import pickle

### Adding Important Paths ###
all_train_files = '../Experiments/context_vector_train_data/'
all_dev_files = '../Experiments/context_vector_dev_data/'
save_autoencoder_model = 'saved_1000/'
model_experiment_name = 'saved_1000'

if not os.path.exists(save_autoencoder_model):
    os.mkdir(save_autoencoder_model)
save_intermediate_results = 'save_temp_files/'
if not os.path.exists(save_intermediate_results):
    os.mkdir(save_intermediate_results)
experiment_data = 'semtab_data,2t_data,limaye_data,biotab_data,t2dv2_data,biodiv_data'
train_files_path = [all_train_files + i for i in experiment_data.split(',')]
dev_files_path = [all_dev_files + i for i in experiment_data.split(',')]
train_files = []
for train_path in train_files_path:
    set_of_files = glob.glob(train_path + '/*.csv')
    train_files.extend(set_of_files)
print(len(train_files))

dev_files = []
for dev_path in dev_files_path:
    set_of_files = glob.glob(dev_path + '/*.csv')
    dev_files.extend(set_of_files)
print(len(dev_files))### Data Preprocessing Code ###

model_save_path = save_autoencoder_model + model_experiment_name
#Include the code to remove duplicates - save everything as text file and then unix sorting
#Merging files
def merge_files(train_files):
    # datapath = args.train_path
    df_list = []
    #kg_id to make sure that duplicates are only removed if same entity
    features = ['context_property_vector', 'kg_id']
    for fn in train_files:
        fid = fn.split('/')[-1][:-4]
        #dataset_id = fn.split('/')[-2]
        df = pd.read_csv(fn)
        if not isinstance(df, pd.DataFrame) :
            continue

        #df['table_id'] = fid
        #df['dataset_id'] = dataset_id
        #df['context_score'].fillna(0.0, inplace=True)
        #if 'column-id' not in df.columns:
        #    df['column-id'] = fn.split('/')[-1] + df['column'].astype('str')

        df = df[features]

        df_list.append(df)
    return pd.concat(df_list)


###Conversion from Dense to Sparse Matrix of alll the properties
# step 1 - calculating all the properties in our dataset
def calculate_all_properties(train_path, dev_path, save_property_list):
    all_train_files_list = glob.glob(train_path + '/*/*.csv')
    print(len(all_train_files_list))
    all_dev_files_list = glob.glob(dev_path + '/*/*.csv')
    print(len(all_dev_files_list))

    properties_set = set()
    inverse_properties = set()
    for file in all_train_files_list:
        df = pd.read_csv(file)
        df['context_property_vector'] = df['context_property_vector'].apply(literal_eval)
        property_values = df['context_property_vector'].values
        for k in property_values:
            for prop in k:
                # print(prop)
                if '_' in prop:
                    inverse_properties.add(prop)
                else:
                    properties_set.add(prop)
    for file in all_dev_files_list:
        df = pd.read_csv(file)
        df['context_property_vector'] = df['context_property_vector'].apply(literal_eval)
        property_values = df['context_property_vector'].values
        for k in property_values:
            for prop in k:
                # print(prop)
                if '_' in prop:
                    inverse_properties.add(prop)
                else:
                    properties_set.add(prop)
    all_properties = properties_set.union(inverse_properties)
    pickle.dump(all_properties, open(save_property_list, 'wb'))

print("Calculating all the properties...........")
save_property_file = save_intermediate_results + 'all_properties.pkl'
if not os.path.exists(save_property_file):
    calculate_all_properties(all_train_files, all_dev_files, save_property_file)
all_properties = pickle.load(open(save_property_file, 'rb'))

properties_list = list(all_properties)
col_indices = {properties_list[i]: i for i in range(len(properties_list))}


def convert_to_matrix_vector(data: pd.DataFrame, properties_list: list):
    # col = properties_list

    col_used_up = set()
    col = list(range(0, len(properties_list)))
    row = list(range(0, len(data)))

    rows, cols, vals = [], [], []
    for rows_ind in range(len(data)):
        if (rows_ind % 5000) == 0:
            print(".......Completed", rows_ind)
        for cols_ind in range(len(data[rows_ind])):
            if isinstance(data[rows_ind][cols_ind], str):
                props = literal_eval(data[rows_ind][cols_ind])
                # print(props)
                for prop in props:
                    rows.append(rows_ind)
                    cols.append(col_indices[prop])
                    col_used_up.add(cols_ind)
                    vals.append(props[prop])
            else:
                pass
    print("----------------------------------------")
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(data), len(properties_list))).toarray()
    return X

print("Converting the dataframe into a sparse matrix.............")
save_matrix_path = save_intermediate_results + 'sparse_matrix.txt'
if not os.path.exists(save_matrix_path):
    print('Merging all the train files')
    # Need to think on whether to include the dev files
    result_df = merge_files(train_files)
    print(result_df.head(10))
    result_df = result_df.drop_duplicates(keep='first')
    sparse_matrix = convert_to_matrix_vector([result_df['context_property_vector'].values.tolist()], all_properties)
    np.savetxt(save_matrix_path, sparse_matrix)
else:
    sparse_matrix = np.loadtxt(save_matrix_path)

print("Sparse Matrix Generated........")
print("Shape of the matrix............", sparse_matrix.shape)
#sparse_matrix_df = pd.DataFrame(sparse_matrix)
### Encoder-Decoder Model Structure ###
class AutoEncoders(Model):

    def __init__(self, layer_1_unit = 1000, layer_2_unit=None, layer_3_unit = None, output_units = None):
        super().__init__()
        self.encoder = Sequential(
            [
                Dense(layer_1_unit, activation="relu"),
                #Dense(layer_2_unit, activation="relu"),
                #Dense(layer_3_unit, activation="relu")
            ]
        )

        self.decoder = Sequential(
            [
                #Dense(layer_2_unit, activation="relu"),
                #Dense(layer_1_unit, activation="relu"),
                Dense(output_units, activation="sigmoid")
            ]
        )


    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded



### Training the model ###
print("Starting the model training ...........")
auto_encoder = AutoEncoders(output_units=len(all_properties))

auto_encoder.compile(
    loss='mae',
    metrics=['mae'],
    optimizer='adam'
)

print("--------Fitting the model ----------")
history = auto_encoder.fit(
    sparse_matrix,
    sparse_matrix,
    epochs=15,
    batch_size=32
)

auto_encoder.save_weights(model_save_path)
auto_encoder.save(save_autoencoder_model + 'model.h5', save_format = 'tf')

### Predicting the model/ using only the encoder layer for the properties ###

#encoder_layer = auto_encoder.get_layer('sequential')
#reduced_df = pd.DataFrame(encoder_layer.predict(x_train_scaled))

################################################################