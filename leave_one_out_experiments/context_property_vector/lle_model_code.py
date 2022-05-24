###########################################################
### Importing Libraries ###
print("Importing Libraries .......")
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import glob
import os
from ast import literal_eval
import scipy.sparse as sp
import pickle
from sklearn.manifold import LocallyLinearEmbedding as LLE
from joblib import dump, load
### CHANGE MODEL CONFIGURATION HERE ###

N_NEIGHBORS = 20
N_DIMS = 50

### Adding Important Paths ###
all_train_files = '../Experiments/context_vector_train_data/'
all_dev_files = '../Experiments/context_vector_dev_data/'
save_autoencoder_model = 'saved_2000_1000/'
model_experiment_name = 'saved_2000_1000'

if not os.path.exists(save_autoencoder_model):
    os.mkdir(save_autoencoder_model)
save_intermediate_results = 'save_temp_files/'
if not os.path.exists(save_intermediate_results):
    os.mkdir(save_intermediate_results)
experiment_data = 'semtab_data,2t_data,limaye_data,t2dv2_data,biodiv_data'
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
    print(len(data))
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
save_matrix_path = save_intermediate_results + 'sparse_matrix_reduced_n_2.txt'
if not os.path.exists(save_matrix_path):
    print('Merging all the train files')
    # Need to think on whether to include the dev files
    result_df = merge_files(train_files)
    print(result_df.head(10))
    result_df = result_df.drop_duplicates(keep='first')
    result_df = result_df.sample(80000)
    print(result_df)
    sparse_matrix_data = [[i] for i in result_df['context_property_vector'].values]
    sparse_matrix = convert_to_matrix_vector(sparse_matrix_data, all_properties)
    np.savetxt(save_matrix_path, sparse_matrix)
else:
    sparse_matrix = np.loadtxt(save_matrix_path)

print("Sparse Matrix Generated........")
print("Shape of the matrix............", sparse_matrix.shape)
#sparse_matrix_df = pd.DataFrame(sparse_matrix)
### LLE CALL CODE / For Modified LLE (Modified is for regularization - mthd = modified)###

def run_lle(num_neighbors, dims, mthd, data):
    # Specify LLE parameters
    embed_lle = LLE(n_neighbors=num_neighbors,  # default=5, number of neighbors to consider for each point.
                    n_components=dims,  # default=2, number of dimensions of the new space
                    reg=0.001,
                    # default=1e-3, regularization constant, multiplies the trace of the local covariance matrix of the distances.
                    eigen_solver='auto',
                    # {‘auto’, ‘arpack’, ‘dense’}, default=’auto’, auto : algorithm will attempt to choose the best method for input data
                    # tol=1e-06, # default=1e-6, Tolerance for ‘arpack’ method. Not used if eigen_solver==’dense’.
                    # max_iter=100, # default=100, maximum number of iterations for the arpack solver. Not used if eigen_solver==’dense’.
                    method=mthd,  # {‘standard’, ‘hessian’, ‘modified’, ‘ltsa’}, default=’standard’
                    # hessian_tol=0.0001, # default=1e-4, Tolerance for Hessian eigenmapping method. Only used if method == 'hessian'
                    modified_tol=1e-12,
                    # default=1e-12, Tolerance for modified LLE method. Only used if method == 'modified'
                    neighbors_algorithm='auto',
                    # {‘auto’, ‘brute’, ‘kd_tree’, ‘ball_tree’}, default=’auto’, algorithm to use for nearest neighbors search, passed to neighbors.NearestNeighbors instance
                    random_state=42,
                    # default=None, Determines the random number generator when eigen_solver == ‘arpack’. Pass an int for reproducible results across multiple function calls.
                    n_jobs=-1  # default=None, The number of parallel jobs to run. -1 means using all processors.
                    )
    # Fit and transofrm the data
    result = embed_lle.fit_transform(data)

    # Return results
    return result


### Training the model ###
print("Starting the model training ...........")
print("Starting the Normal LLE...")
std_lle_res=run_lle(num_neighbors=N_NEIGHBORS, dims=N_DIMS, mthd='standard', data=sparse_matrix)

# Modified LLE
print("Starting a modified LLE...")
#mlle_res=run_lle(num_neighbors=N_NEIGHBORS, dims=N_DIMS, mthd='modified', data=sparse_matrix)

print('MODEL FIT COMPLETE! Saving the models ...')
dump(std_lle_res, f'lle_neigh_{N_NEIGHBORS}_{N_DIMS}D.bin', compress=True)
#dump(mlle_res, f'mlle_neigh_{N_NEIGHBORS}_{N_DIMS}D.bin', compress=True)
# To load the model
#sc=load('std_scaler.bin')
################################################################