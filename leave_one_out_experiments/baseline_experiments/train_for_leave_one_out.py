# Purpose:
#For leave one out experiments, to set the baseline scores, we give a simple run over all datasets with the leave one out strategy: 
# train and dev: all datasets except test. This script allows to set up an experiment over a particular dataset and runs the model with different shuffling strategy and hyperparameter values. 

# Set up particular dataset and train dataset in Defining Parameters

# For data: data is downloaded directly with the AWS S3 during the model running. Define your Access Key and ID in Defining connection to S3
# Alternatively create dataset using feature_generation.py


###Importing all the libraries
import glob
import boto3
import time
import os
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
import pickle
from argparse import ArgumentParser, Namespace
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from itertools import chain
import copy
import shutil
import pickle
from tqdm import tqdm

# Defining connection to s3
ACCESS_ID = # Your access id here
ACCESS_KEY = # Your access key here

s3 = boto3.client('s3', aws_access_key_id=ACCESS_ID,
                  aws_secret_access_key=ACCESS_KEY, use_ssl=True)
bucket = 'table-linker-datasets'
# Defining parameters
experiment_name = "Experiment_test_biotab"
experiment_data = 'biodiv_data,semtab_data,limaye_data,t2dv2_data,2t_data'
features = ["monge_elkan","monge_elkan_aliases","jaro_winkler",
            "levenshtein","singleton","context_score_3","pgt_centroid_score","pgt_class_count_tf_idf_score",
            "pgt_property_count_tf_idf_score", "num_occurences", "incorrectness_scores"]
s3_1 = boto3.resource("s3", aws_access_key_id=ACCESS_ID,
                      aws_secret_access_key=ACCESS_KEY)
bucket = "table-linker-datasets"
folder = "Experiments/reduced_train_data"
s3_bucket = s3_1.Bucket(bucket)

### Creating the directories for the results.
experiment_train_data = experiment_data
experiment_dev_data =  experiment_data
experiment_store_path = f"Experiments/{experiment_name}"
dev_predictions = f"{experiment_store_path}/dev/dev_predictions/"
dev_output_pred = f"{experiment_store_path}/dev/dev_output/"
dev_predictions_top_k = f"{experiment_store_path}/dev/dev_predictions_top_k/"
dev_metrics = f"{experiment_store_path}/dev/dev_metrics/"
dev_predictions_colorized = f"{experiment_store_path}/dev/dev_predictions_colorized/"
model_save_path = f'{experiment_store_path}/final_models/'
best_model_path = ''
# Need to append the version after this so as to represent the learning rate, batch and shuffle theory
# One version would be somthing like v1...v10 and named like v1_lr_0001_bs_32_shuffle_no_shuffle
training_data_path = f'{experiment_store_path}/model_training_data'
# get_ipython().system('mkdir -p $training_data_path')
pos_output = f'{training_data_path}/tl_pipeline_pos_features.pkl'
neg_output = f'{training_data_path}/tl_pipeline_neg_features.pkl'
min_max_scaler_path = f'{training_data_path}/tl_pipeline_normalization_factor.pkl'

final_score_column = 'siamese_prediction'
threshold = final_score_column + ":median"
train_files_path = [f'Experiments/{experiment_name}/reduced_train_data/' + i for i in experiment_train_data.split(',')]
extra_feat = ['column-id', 'column', 'row', 'evaluation_label']
for f in features:
    extra_feat.append(f)
final_score_column = 'siamese_prediction'
threshold = final_score_column + ":median"
dev_files_path = ['Experiments/reduced_dev_data/' + i for i in experiment_train_data.split(',')]
learning_rate_changes = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
batch_size_changes = [32, 16, 8]
shuffling_strategy = ["cell", "table", "dataset","complete_shuffle"]
print("Started finding the dev files")
dev_files_path =[f'Experiments/{experiment_name}/reduced_dev_data/' + i for i in experiment_train_data.split(',')]
dev_files = []
for dev_path in dev_files_path:
    set_of_files = [i.key for i in s3_bucket.objects.filter(Prefix=dev_path).all()]
    dev_files.extend(set_of_files[1:])
print(len(dev_files))
details = []
batch_size = 32
learning_rate = 0.00001


def read_file(key):
    resp = s3.get_object(Bucket = bucket, Key = key)
    try:
        df = pd.read_csv(resp['Body'], sep = ',')
    except pd.errors.EmptyDataError:
        new_key = key.replace(experiment_name + '/', '')
        resp_2 = s3.get_object(Bucket=bucket, Key=key)
        try:
            df = pd.read_csv(resp['Body'], sep = ',')
        except:
            df = ''
        print('Empty csv file!')
    return df
'''
def read_file(key):
    resp = s3.get_object(Bucket = bucket, Key = key)
    try:
        df = pd.read_csv(resp['Body'], sep = ',')
    except pd.errors.EmptyDataError:
        df = ''
        print('Empty csv file!')
    return df
'''
def save_file(key, content):
    s3_res = boto3.resource('s3')

    object = s3_res.Object(bucket, key)
    result = object.put(Body = content)

def merge_files(args):
    # datapath = args.train_path
    df_list = []
    for fn in args.train_files:
        fid = fn.split('/')[-1][:-4]
        dataset_id = fn.split('/')[-2]
        df = read_file(fn)
        if not isinstance(df, pd.DataFrame) :
            continue

        df['table_id'] = fid
        df['dataset_id'] = dataset_id
        df['context_score'].fillna(0.0, inplace=True)
        if 'column-id' not in df.columns:
            df['column-id'] = fn.split('/')[-1] + df['column'].astype('str')

        df = df[extra_feat]
        df_list.append(df)
    return pd.concat(df_list)


def compute_normalization_factor(args, all_data):
    min_max_scaler_path = args.min_max_scaler_path
    all_data_features = all_data[features]
    scaler = MinMaxScaler()
    scaler.fit(all_data_features)
    pickle.dump(scaler, open('./tmp/min_max_scaler_path.pkl', 'wb'))
    s3_1.Bucket('table-linker-datasets').upload_file('./tmp/min_max_scaler_path.pkl', min_max_scaler_path)

    #save_file(min_max_scaler_path, scaler)
    return scaler

def save_pickle(key, content):
    pass

# In[10]:


def generate_train_data(args, all_data, shuffle_by = None):
    num_cells_1 = 0
    scaler_path = args.min_max_scaler_path
    scaler = pickle.load(open('./tmp/min_max_scaler_path.pkl', 'rb'))
    final_list = []
    sfeatures = copy.deepcopy(features) + ['evaluation_label']
    normalize_features = features
    evaluation_label = ['evaluation_label']
    positive_features_final = []
    negative_features_final = []
    super_groups = all_data.groupby(['column-id'])
    if shuffle_by == 'dataset':
        super_groups = all_data.groupby(['dataset_id'])
        for i, s_group in super_groups:
            pos_features_dataset = []
            neg_features_dataset = []
            grouped_obj = s_group.groupby(['column', 'row', 'column-id'])

            for cell in grouped_obj:
                num_cells_1 += 1
                cell[1][normalize_features] = scaler.transform(cell[1][normalize_features])
                pos_features = []
                neg_features = []
                a = cell[1][cell[1]['evaluation_label'] == 1]
                if a.empty:
                    continue
                pos_rows = cell[1][cell[1]['evaluation_label'].astype(int) == 1][features].to_numpy()
                if len(pos_rows) < 1:
                    continue
                if len(pos_rows) > 1:
                    print("here")
                for i in range(len(pos_rows)):
                    pos_features.append(pos_rows[i])
                neg_rows = cell[1][cell[1]['evaluation_label'].astype(int) == -1][features].to_numpy()
                for i in range(min(batch_size, len(neg_rows))):
                    neg_features.append(neg_rows[i])

                for k in range(len(neg_features) - len(pos_features)):
                    pos_features.append(random.choice(pos_rows))
                if len(pos_features) != len(neg_features):
                    continue
                pos_features_dataset.append(pos_features)
                neg_features_dataset.append(neg_features)
        if len(pos_features_table) > 0:
            c = list(zip(pos_features_dataset, neg_features_dataset))
            random.shuffle(c)
            pos_features_dataset, neg_features_dataset = zip(*c)
            positive_features_final.extend(pos_features_dataset)
            negative_features_final.extend(neg_features_dataset)

    else:
        for i, s_group in super_groups:
            file_name = i.split('-')[0]
            #print("entering ", file_name)
            grouped_obj = s_group.groupby(['column', 'row'])
            if shuffle_by == 'cell':
                for cell in grouped_obj:
                    num_cells_1 += 1
                    cell[1][normalize_features] = scaler.transform(cell[1][normalize_features])
                    pos_features = []
                    neg_features = []
                    a = cell[1][cell[1]['evaluation_label'] == 1]
                    if a.empty:
                        continue
                    pos_rows = cell[1][cell[1]['evaluation_label'].astype(int) == 1][features].to_numpy()
                    if len(pos_rows) < 1:
                        continue
                    if len(pos_rows) > 1:
                        print("here")
                    for i in range(len(pos_rows)):
                        pos_features.append(pos_rows[i])
                    neg_rows = cell[1][cell[1]['evaluation_label'].astype(int) == -1][features].to_numpy()
                    for i in range(min(batch_size, len(neg_rows))):
                        neg_features.append(neg_rows[i])

                    for k in range(len(neg_features) - len(pos_features)):
                        pos_features.append(random.choice(pos_rows))
                    random.shuffle(pos_features)
                    random.shuffle(neg_features)
                    if len(pos_features) != len(neg_features):
                        print("HHHERRRERR")
                    else:
                        positive_features_final.append(pos_features)
                        negative_features_final.append(neg_features)
                print(len(positive_features_final), len(positive_features_final[3]))
                print(len(negative_features_final), len(negative_features_final[3]))

            elif shuffle_by == 'table':
                pos_features_table = []
                neg_features_table = []
                for cell in grouped_obj:
                    num_cells_1 += 1
                    cell[1][normalize_features] = scaler.transform(cell[1][normalize_features])
                    pos_features = []
                    neg_features = []
                    a = cell[1][cell[1]['evaluation_label'] == 1]
                    if a.empty:
                        continue
                    pos_rows = cell[1][cell[1]['evaluation_label'].astype(int) == 1][features].to_numpy()
                    if len(pos_rows) < 1:
                        continue
                    if len(pos_rows) > 1:
                        print("here")
                    for i in range(len(pos_rows)):
                        pos_features.append(pos_rows[i])
                    neg_rows = cell[1][cell[1]['evaluation_label'].astype(int) == -1][features].to_numpy()
                    for i in range(min(batch_size, len(neg_rows))):
                        neg_features.append(neg_rows[i])

                    for k in range(len(neg_features) - len(pos_features)):
                        pos_features.append(random.choice(pos_rows))
                    if len(pos_features) != len(neg_features):
                        continue
                    random.shuffle(pos_features)

                    random.shuffle(neg_features)
                    pos_features_table.append(pos_features)
                    neg_features_table.append(neg_features)
                if len(pos_features_table) > 0:
                    c = list(zip(pos_features_table, neg_features_table))
                    random.shuffle(c)
                    pos_features_table, neg_features_table = zip(*c)
                    positive_features_final.extend(pos_features_table)
                    negative_features_final.extend(neg_features_table)

            elif shuffle_by == 'complete_shuffle':
                for cell in grouped_obj:
                    num_cells_1 += 1
                    cell[1][normalize_features] = scaler.transform(cell[1][normalize_features])
                    pos_features = []
                    neg_features = []
                    a = cell[1][cell[1]['evaluation_label'] == 1]
                    if a.empty:
                        continue
                    pos_rows = cell[1][cell[1]['evaluation_label'].astype(int) == 1][features].to_numpy()
                    if len(pos_rows) < 1:
                        continue
                    if len(pos_rows) > 1:
                        print("here")
                    for i in range(len(pos_rows)):
                        pos_features.append(pos_rows[i])
                    neg_rows = cell[1][cell[1]['evaluation_label'].astype(int) == -1][features].to_numpy()
                    for i in range(min(batch_size, len(neg_rows))):
                        neg_features.append(neg_rows[i])

                    for k in range(len(neg_features) - len(pos_features)):
                        pos_features.append(random.choice(pos_rows))
                    random.shuffle(pos_features)
                    random.shuffle(neg_features)
                    if len(pos_features) != len(neg_features):
                        print("HHHERRRERR")
                    else:
                        positive_features_final.append(pos_features)
                        negative_features_final.append(neg_features)
                c = list(zip(negative_features_final, positive_features_final))
                random.shuffle(c)
                positive_features_final, positive_features_final = zip(*c)
                print(len(positive_features_final), len(positive_features_final[3]))
                print(len(negative_features_final), len(negative_features_final[3]))
    pickle.dump(positive_features_final, open('./tmp/pos.pkl', 'wb'))
    pickle.dump(negative_features_final, open('./tmp/neg.pkl', 'wb'))
    print(len(positive_features_final), len(positive_features_final[3]))
    print(len(negative_features_final), len(negative_features_final[3]))

    #save_file(args.pos_output, positive_features_final)
    #save_file(args.neg_output, negative_features_final)
    s3_1.Bucket('table-linker-datasets').upload_file('./tmp/pos.pkl', args.pos_output)
    s3_1.Bucket('table-linker-datasets').upload_file('./tmp/neg.pkl', args.neg_output)

class T2DV2Dataset(Dataset):
    def __init__(self, pos_features, neg_features):
        self.pos_features = pos_features
        self.neg_features = neg_features

    def __len__(self):
        return len(self.pos_features)

    def __getitem__(self, idx):
        return self.pos_features[idx], self.neg_features[idx]


# Model
class PairwiseNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # original 12x24, 24x12, 12x12, 12x1
        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, pos_features, neg_features):
        # Positive pass
        x = F.relu(self.fc1(pos_features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        pos_out = torch.sigmoid(self.fc4(x))

        # Negative Pass
        x = F.relu(self.fc1(neg_features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        neg_out = torch.sigmoid(self.fc4(x))

        return pos_out, neg_out

    def predict(self, test_feat):
        x = F.relu(self.fc1(test_feat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        test_out = torch.sigmoid(self.fc4(x))
        return test_out


# Pairwise Loss
class PairwiseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = 0

    def forward(self, pos_out, neg_out):
        distance = (1 - pos_out) + neg_out
        loss = torch.mean(torch.max(torch.tensor(0), distance))
        return loss


def generate_dataloader(positive_feat_path, negative_feat_path):
    pos_features = pickle.loads(s3_1.Bucket("table-linker-datasets").Object(positive_feat_path).get()['Body'].read())
    neg_features = pickle.loads(s3_1.Bucket("table-linker-datasets").Object(negative_feat_path).get()['Body'].read())
    # pos_features = pickle.load(open(positive_feat_path, 'rb'))
    # neg_features = pickle.load(open(negative_feat_path, 'rb'))

    pos_features_flatten = list(chain.from_iterable(pos_features))
    neg_features_flatten = list(chain.from_iterable(neg_features))
    print(len(pos_features_flatten), len(neg_features_flatten), pos_features_flatten[0], neg_features_flatten[0])
    # print(len(pos_features_flatten), len(neg_features_flatten))
    train_dataset = T2DV2Dataset(pos_features_flatten, neg_features_flatten)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    return train_dataloader


def infer_scores(min_max_scaler_path, input_table_path, output_table_path, model):
    scaler = pickle.loads(s3_1.Bucket("table-linker-datasets").Object(min_max_scaler_path).get()['Body'].read())
    normalize_features = features
    number_of_cells_top_1 = 0
    number_of_cells_total = 0
    for file in input_table_path:
        file_name = file.split('/')[-1]

        d_sample = read_file(file)
        if not isinstance(d_sample, pd.DataFrame):
            continue
        grouped_obj = d_sample.groupby(['column', 'row'])
        new_df_list = []
        pred = []
        for cell in grouped_obj:
            cell[1][normalize_features] = scaler.transform(cell[1][normalize_features])
            sorted_df = cell[1].sort_values('context_score', ascending=False)
            sorted_df_features = sorted_df[normalize_features]
            new_df_list.append(sorted_df)
            arr = sorted_df_features.to_numpy()
            test_inp = []
            for a in arr:
                test_inp.append(a)
            test_tensor = torch.tensor(test_inp).float()
            scores = model.predict(test_tensor)
            scores_list = torch.squeeze(scores).tolist()
            if not type(scores_list) is list:
                pred.append(scores_list)
            else:
                pred.extend(scores_list)
        test_df = pd.concat(new_df_list)
        test_df[final_score_column] = pred
        test_df['table_id'] = file_name
        # df_input_table.append(test_df)
        num_of_cells_with_correct_top_1, num_of_cells = parse_eval_files_stats(test_df, 'siamese_prediction')
        number_of_cells_top_1 += num_of_cells_with_correct_top_1
        number_of_cells_total += num_of_cells
    print(number_of_cells_top_1, number_of_cells_total, input_table_path)
    return number_of_cells_top_1 / number_of_cells_total


def train(args, train_dataloader_1):
    if torch.cuda.is_available():
        device = torch.device('cuda')

    else:
        device = torch.device('cpu')
    device = torch.device('cpu')

    criterion = PairwiseLoss()
    EPOCHS = args.num_epochs
    model = PairwiseNetwork(len(features)).to(device=device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
    top1_max_prec = 0
    for epoch in range(EPOCHS):
        train_epoch_loss = 0
        avg_loss = 0
        model.train()
        for bid, batch in enumerate(train_dataloader_1):
            # print("--------------")
            positive_feat = torch.tensor(batch[0].float())
            negative_feat = torch.tensor(batch[1].float())
            optimizer.zero_grad()
            # print(positive_feat.is_cuda, negative_feat.is_cuda)
            pos_out, neg_out = model(positive_feat, negative_feat)
            ##print(pos_out.is_cuda, neg_out.is_cuda, model.is_cuda)
            loss = criterion(pos_out, neg_out)
            # print(loss.is_cuda)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss
        avg_loss = train_epoch_loss / bid
        # scheduler.step()
        # Evaluation
        model.eval()
        top1_precision = infer_scores(args.min_max_scaler_path, args.dev_path, args.dev_output, model)
        # eval_data = merge_eval_files(args.dev_output)
        # res, candidate_eval_data = parse_eval_files_stats(eval_data, final_score_column)
        # top1_precision = res['num_tasks_with_model_score_top_one_accurate'] / res['num_tasks_with_gt']
        if top1_precision > top1_max_prec:
            # copy_files(args.dev_output, dev_predictions)
            top1_max_prec = top1_precision
            model_save_name = 'top1_{}_epoch_{}_loss_{}_batch_size_{}_learning_rate_{}.pth'.format(top1_max_prec, epoch,
                                                                                                   avg_loss, batch_size,
                                                                                                   learning_rate)
            best_model_path = args.model_save_path + model_save_name
            torch.save(model.state_dict(), '/tmp/' + model_save_name)
        s3_1.Bucket('table-linker-datasets').upload_file('/tmp/' + model_save_name, best_model_path)

        print("Epoch {}, Avg Loss is {}, epoch top1 {}, max top1 {}".format(epoch, avg_loss, top1_precision,
                                                                            top1_max_prec))
    return best_model_path, top1_max_prec


def merge_eval_files(final_score_path):
    df_list = []
    for fn in final_score_path:
        fid = fn.split('/')[-1].split('.csv')[0]
        df = pd.read_csv(fn)
        if not isinstance(df, pd.DataFrame):
            continue
        df['table_id'] = fid
        df_list.append(df)
    return pd.concat(df_list)


def parse_eval_files_stats(eval_data, method):
    res = {}
    candidate_eval_data = eval_data.groupby(['table_id', 'column', 'row'])['table_id'].count().reset_index(name="count")
    res['num_tasks_with_gt'] = len(eval_data[pd.notna(eval_data['GT_kg_id'])].groupby(['table_id', 'column', 'row']))
    num_tasks_with_model_score_top_one_accurate = []
    num_tasks_with_model_score_top_five_accurate = []
    num_tasks_with_model_score_top_ten_accurate = []
    has_gt_list = []
    has_gt_in_candidate = []
    for i, row in candidate_eval_data.iterrows():
        table_id, row_idx, col_idx = row['table_id'], row['row'], row['column']
        c_e_data = eval_data[
            (eval_data['table_id'] == table_id) & (eval_data['row'] == row_idx) & (eval_data['column'] == col_idx)]
        assert len(c_e_data) > 0
        if np.nan not in set(c_e_data['GT_kg_id']):
            has_gt_list.append(1)
        else:
            has_gt_list.append(0)
        if 1 in set(c_e_data['evaluation_label']):
            has_gt_in_candidate.append(1)
        else:
            has_gt_in_candidate.append(0)

        # rank on model score
        s_data = c_e_data.sort_values(by=[method], ascending=False)
        if s_data.iloc[0]['evaluation_label'] == 1:
            num_tasks_with_model_score_top_one_accurate.append(1)
        else:
            num_tasks_with_model_score_top_one_accurate.append(0)

    res['num_tasks_with_model_score_top_one_accurate'] = sum(num_tasks_with_model_score_top_one_accurate)
    # print(sum(num_tasks_with_model_score_top_one_accurate))
    return res['num_tasks_with_model_score_top_one_accurate'], res['num_tasks_with_gt']
# In[11]:


for shuff_method in shuffling_strategy:
    model_save_path = f'{experiment_store_path}/final_models/{shuff_method}/'
    pos_output = f'{training_data_path}/{shuff_method}_tl_pipeline_pos_features.pkl'
    neg_output = f'{training_data_path}/{shuff_method}_tl_pipeline_neg_features.pkl'
    min_max_scaler_path = f'{training_data_path}/tl_pipeline_normalization_factor.pkl'
    training_args = Namespace(num_epochs=3, lr=learning_rate, positive_feat_path=pos_output,
                                  negative_feat_path=neg_output,
                                  dev_path=dev_files, dev_output=dev_files,
                                  model_save_path=model_save_path, min_max_scaler_path=min_max_scaler_path)
    train_dataloader_1 = generate_dataloader(pos_output, neg_output)
    best_model_path, precision = train(training_args, train_dataloader_1)
    print(precision)
    details.append([batch_size, shuff_method, learning_rate, precision])
    print(details)
