# Helper functions: Purpose: Uses the context similarity algorithm to convert context into context property vector and finally to context_property similarity vector when called by the actual notebook.
import pandas as pd
import glob
import numpy as np
import pickle
from ast import literal_eval
import scipy.sparse as sp
import os
from sklearn.decomposition import PCA

input_location = ['../Experiments/context_vector_train_data/', '../Experiments/context_vector_dev_data/']
result_locations = ['../Experiments/context_vector_train_data/context_similarity_vector/',
'../Experiments/context_vector_dev_data/context_similarity_vector/']
datasets = 'semtab_data,2t_data,limaye_data,biodiv_data,biotab_data'

all_properties = pickle.load(open('all_properties.pkl', 'rb'))

properties_list = list(all_properties)
# col_indices_1 = {i:i for i in range(len(features))}

col_indices = {properties_list[i]: i for i in range(len(properties_list))}


def convert_to_matrix_vector(data: list, properties_list: list, autoencoder_layer = None):
    # col = properties_list

    col_used_up = set()
    col = list(range(0, len(properties_list)))
    row = list(range(0, len(data)))

    # data['context_property_vector'] = data['context_property_vector'].apply(literal_eval)
    # property_values = df['context_property_vector'].values
    rows, cols, vals = [], [], []
    # print(len(properties_list))
    features_vals = []
    for rows_ind in range(len(data)):
        feature_range = []
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
                feature_range.append(data[rows_ind][cols_ind])
        features_vals.append(feature_range)
    Y = np.array(features_vals)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(data), len(properties_list))).toarray()
    if autoencoder_layer is not None:
        X = autoencoder_layer.predict(X)
    complete = np.concatenate((Y, X), axis=1)
    # print(complete[0])
    return complete


# Save the individual files as pickle


# In[33]:
def create_context_similarity(df, properties_list = properties_list, consider_positives_only = False, multiple_column = True, apply_pca_components=None, concatenate = False, autoencoder = None, return_context_property=False):
    unique_columns = df['column'].unique().tolist()
    result_array = []
    if consider_positives_only:
        df = df[df['evaluation_label'] == 1]
    if multiple_column:
        for col in unique_columns:
            data = df[df['column'] == col]
            context_property_vector = convert_to_matrix_vector([[i] for i in data['context_property_vector'].values], properties_list, autoencoder)
            context_property_vector_t = np.transpose(context_property_vector)
            context_property_similarity = np.matmul(context_property_vector, context_property_vector_t)
            if apply_pca_components:
                pca_kernel = PCA(n_components = apply_pca_components)
                context_property_similarity = pca_kernel.fit_transform(context_property_similarity)
            result_array.append(context_property_similarity)
    else:
        context_property_vector = convert_to_matrix_vector([[i] for i in df['context_property_vector'].values], properties_list, autoencoder)
        context_property_vector_t = np.transpose(context_property_vector)
        context_property_similarity = np.matmul(context_property_vector, context_property_vector_t)
        if apply_pca_components:
            pca_kernel = PCA(n_components = apply_pca_components)
            context_property_similarity = pca_kernel.fit_transform(context_property_similarity)
        result_array.append(context_property_similarity)
    result_array = np.array(result_array, dtype = 'object')
    if concatenate:
        result_array = np.concatenate(result_array, axis = 0)
    if return_context_property:
        return [result_array, context_property_vector]
    else:
        return [result_array, None]

def run_over_all_data():
    for ind,result_location in enumerate(result_locations):
        get_ipython().system('mkdir -p $result_location')
        for dataset in datasets.split(','):
            dataset_result_location = result_location + dataset
            get_ipython().system('mkdir -p $dataset_result_location')
            dataset_input_location = input_location[ind] + dataset
            for file in glob.glob(dataset_input_location + '/*.csv'):
                filename = file.split('/')[-1]
                final_result_path = f'{dataset_result_location}/{filename[:-4]}.pkl'
                print(file, final_result_path)
                if os.path.exists(final_result_path):
                    continue
                df = pd.read_csv(file)
                unique_columns = df['column'].unique().tolist()
                result_array = create_context_similarity(df)
                pickle.dump(result_array, open(final_result_path, 'wb'))
                
                
def return_a_table(df, pca_components = None, result_path = None, autoencoder = None, return_context_property = False):
    #df = pd.read_csv(table_path)
    context_similarity_result_array = create_context_similarity(df, apply_pca_components = pca_components, concatenate=True, return_context_property= return_context_property, autoencoder = autoencoder)
    
    context_similarity_result_df = pd.DataFrame(context_similarity_result_array[0])
    context_similarity_result_df.columns = [f'csp_{i}' for i in range(pca_components)]
    if len(context_similarity_result_df) != len(df):
        print("Something Wrong !")
    if return_context_property:
        context_property_vec_df = pd.DataFrame(context_similarity_result_array[1])
        context_property_vec_df.columns = [f'cpv_{i}' for i in range(context_property_vec_df.shape[1])]
        result_df = pd.concat([df, context_similarity_result_df, context_property_vec_df], axis = 1)
    else:
        result_df = pd.concat([df, context_similarity_result_df], axis = 1)
    if result_path:
        result_df.to_csv(result_path, index = None)

    return result_df
    

# In[10]: