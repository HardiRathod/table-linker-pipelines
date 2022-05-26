# Purpose: This is an example files for biodiv as test data and other datasets as train and dev set. It converts the originial data into postive negative pair samples and saves as pickle file after normalization. TO train the pseudo ground truth model, it reloads it, and runs to train the model. 

# You can set parameters like experiment_name, test and train datasets, learning rate batch size and shuffling startegy. 

# Data is directly downloaded from AWS, Optionally set local path.

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
s3 = boto3.client('s3')
bucket = 'table-linker-datasets'
###Defining all the results path and version controls


## Final Test and Train Results need to be stored in the same directory as Experiment.

#get_ipython().system('mkdir -p $model_save_path')


### Functions needed to train the files.
def read_file(key):
    #resp = s3.get_object(Bucket = bucket, Key = key)
    try:
        df = pd.read_csv(key)
    except pd.errors.EmptyDataError:
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
        df = read_file(fn)
        if not isinstance(df, pd.DataFrame) :
            continue

        df['table_id'] = fid
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


def generate_train_data(args, all_data):
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
    for i, s_group in super_groups:
        file_name = i.split('-')[0]
        #print("entering ", file_name)
        grouped_obj = s_group.groupby(['column', 'row'])

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
    # print(num_cells_1)
    pickle.dump(positive_features_final, open('./tmp/pos.pkl', 'wb'))
    pickle.dump(negative_features_final, open('./tmp/neg.pkl', 'wb'))
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
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
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
    pos_features = pickle.load(open(positive_feat_path, 'rb'))
    neg_features = pickle.load(open(negative_feat_path, 'rb'))

    pos_features_flatten = list(chain.from_iterable(pos_features))
    neg_features_flatten = list(chain.from_iterable(neg_features))
    print(len(pos_features_flatten), len(neg_features_flatten), pos_features_flatten[0], neg_features_flatten[0])
    # print(len(pos_features_flatten), len(neg_features_flatten))
    train_dataset = T2DV2Dataset(pos_features_flatten, neg_features_flatten)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    return train_dataloader


def infer_scores(min_max_scaler_path, input_table_path, output_table_path, model):
    scaler = pickle.load(open(min_max_scaler_path, 'rb'))
    normalize_features = features
    for file in input_table_path:
        file_name = file.split('/')[-1]

        d_sample = read_file(file)
        if not isinstance(d_sample,pd.DataFrame):
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
        test_df.to_csv(f"{output_table_path}/{file_name}", index=False)


def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')

    else:
        device = torch.device('cpu')
    device = torch.device('cpu')
    train_dataloader_1 = generate_dataloader(args.positive_feat_path, args.negative_feat_path)
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
        for bid_1, batch_1 in tqdm(enumerate(train_dataloader_1), position=0, leave=True):
            # print("--------------")
            positive_feat = torch.tensor(batch_1[0].float())
            negative_feat = torch.tensor(batch_1[1].float())
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
        infer_scores(args.min_max_scaler_path, args.dev_path, args.dev_output, model)
        eval_data = merge_eval_files(args.dev_output)
        res, candidate_eval_data = parse_eval_files_stats(eval_data, final_score_column)
        top1_precision = res['num_tasks_with_model_score_top_one_accurate'] / res['num_tasks_with_gt']
        if top1_precision > top1_max_prec:
            #copy_files(args.dev_output, dev_predictions)
            top1_max_prec = top1_precision
            model_save_name = 'top1_{}_epoch_{}_loss_{}_batch_size_{}_learning_rate_{}.pth'.format(top1_max_prec, epoch,
                                                                                                   avg_loss, batch_size,
                                                                                                   learning_rate)
            best_model_path = args.model_save_path + model_save_name
            torch.save(model.state_dict(), '/tmp/'+model_save_name)
        s3_1.Bucket('table-linker-datasets').upload_file('/tmp/'+model_save_name, best_model_path)

        print("Epoch {}, Avg Loss is {}, epoch top1 {}, max top1 {}".format(epoch, avg_loss, top1_precision,
                                                                            top1_max_prec))
    return best_model_path


def merge_eval_files(final_score_path):

    for fn in final_score_path:
        fid = fn.split('/')[-1].split('.csv')[0]
        df = read_file(fn)
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
        if 1 in set(s_data.iloc[0:5]['evaluation_label']):
            num_tasks_with_model_score_top_five_accurate.append(1)
        else:
            num_tasks_with_model_score_top_five_accurate.append(0)
        if 1 in set(s_data.iloc[0:10]['evaluation_label']):
            num_tasks_with_model_score_top_ten_accurate.append(1)
        else:
            num_tasks_with_model_score_top_ten_accurate.append(0)

    res['num_tasks_with_model_score_top_one_accurate'] = sum(num_tasks_with_model_score_top_one_accurate)
    res['num_tasks_with_model_score_top_five_accurate'] = sum(num_tasks_with_model_score_top_five_accurate)
    res['num_tasks_with_model_score_top_ten_accurate'] = sum(num_tasks_with_model_score_top_ten_accurate)
    return res, candidate_eval_data

# In[11]:
batch_size = 32
learning_rate = 0.0001
s3_1 = boto3.resource("s3")
bucket = "table-linker-datasets"
folder = "Experiments/reduced_train_data"
s3_bucket = s3_1.Bucket(bucket)
#files_in_s3 = [f.key for f in s3_bucket.objects.filter(Prefix=folder).all()]
features = ["monge_elkan", "monge_elkan_aliases", "jaro_winkler", "levenshtein", "singleton", "pgr_rts",
            "context_score", "smc_class_score", "smc_property_score"]
experiment_name = "Experiment_test_biodiv"
experiment_data = '2t_data,semtab_data,t2dv2_data,biotab_data,limaye_data'
### Creating the directories for the results.
experiment_train_data = experiment_data
experiment_dev_data =  experiment_data
experiment_store_path = f"Experiments/{experiment_name}"
dev_predictions = f"{experiment_store_path}/dev/dev_predictions/"
dev_output_pred = f"{experiment_store_path}/dev/dev_output/"
dev_predictions_top_k = f"{experiment_store_path}/dev/dev_predictions_top_k/"
dev_metrics = f"{experiment_store_path}/dev/dev_metrics/"
dev_predictions_colorized = f"{experiment_store_path}/dev/dev_predictions_colorized/"
model_save_path = f'{experiment_store_path}/pgt_models/'
best_model_path = ''
# Need to append the version after this so as to represent the learning rate, batch and shuffle theory
# One version would be somthing like v1...v10 and named like v1_lr_0001_bs_32_shuffle_no_shuffle
training_data_path = f'{experiment_store_path}/training_data'
# get_ipython().system('mkdir -p $training_data_path')
pos_output = f'{training_data_path}/tl_pipeline_pos_features.pkl'
neg_output = f'{training_data_path}/tl_pipeline_neg_features.pkl'
min_max_scaler_path = f'{training_data_path}/pseudo_pipeline_normalization_factor.pkl'

final_score_column = 'siamese_prediction'
threshold = final_score_column + ":median"
train_files_path = ['Experiments/reduced_train_data/' + i for i in experiment_train_data.split(',')]
extra_feat = ['column-id', 'column', 'row', 'evaluation_label']
for f in features:
    extra_feat.append(f)
def handler(event, context):



    global experiment_name
    global experiment_data
    ### Creating the directories for the results.
    global experiment_train_data
    global experiment_dev_data
    global experiment_store_path
    global dev_predictions
    global dev_output_pred
    global dev_predictions_top_k
    global dev_metrics
    global dev_predictions_colorized
    global model_save_path
    #best_model_path = ''
    # Need to append the version after this so as to represent the learning rate, batch and shuffle theory
    # One version would be somthing like v1...v10 and named like v1_lr_0001_bs_32_shuffle_no_shuffle
    global training_data_path
    # get_ipython().system('mkdir -p $training_data_path')
    global pos_output
    global neg_output
    global min_max_scaler_path
    experiment_name = event['Experiment_name']
    experiment_data = event['Experiment_data']
    final_score_column = 'siamese_prediction'
    threshold = final_score_column + ":median"
    train_files_path = ['Experiments/reduced_train_data/' + i for i in experiment_train_data.split(',')]
    ### Creating the directories for the results.
    experiment_train_data = experiment_data
    experiment_dev_data = experiment_data
    experiment_store_path = f"Experiments/{experiment_name}"
    dev_predictions = f"{experiment_store_path}/dev/dev_predictions/"
    dev_output_pred = f"{experiment_store_path}/dev/dev_output/"
    dev_predictions_top_k = f"{experiment_store_path}/dev/dev_predictions_top_k/"
    dev_metrics = f"{experiment_store_path}/dev/dev_metrics/"
    dev_predictions_colorized = f"{experiment_store_path}/dev/dev_predictions_colorized/"
    model_save_path = f'{experiment_store_path}/pgt_models/'
    best_model_path = ''
    # Need to append the version after this so as to represent the learning rate, batch and shuffle theory
    # One version would be somthing like v1...v10 and named like v1_lr_0001_bs_32_shuffle_no_shuffle
    training_data_path = f'{experiment_store_path}/training_data'
    # get_ipython().system('mkdir -p $training_data_path')
    pos_output = f'{training_data_path}/tl_pipeline_pos_features.pkl'
    neg_output = f'{training_data_path}/tl_pipeline_neg_features.pkl'
    min_max_scaler_path = f'{training_data_path}/pseudo_pipeline_normalization_factor.pkl'

final_score_column = 'siamese_prediction'
threshold = final_score_column + ":median"
train_files_path = ['Experiments/reduced_train_data/' + i for i in experiment_train_data.split(',')]
dev_files_path = ['Experiments/dev_data/' + i for i in experiment_train_data.split(',')]



train_files = []
for train_path in train_files_path:
    set_of_files = [i.key for i in s3_bucket.objects.filter(Prefix=train_path).all()]
    train_files.extend(set_of_files[1:])
dev_files = []
for dev_path in dev_files_path:
    set_of_files = [i.key for i in s3_bucket.objects.filter(Prefix=dev_path).all()]
    dev_files.extend(set_of_files[1:])
gen_training_data_args = Namespace(train_files=train_files, pos_output=pos_output, neg_output=neg_output,min_max_scaler_path=min_max_scaler_path)
all_data = merge_files(gen_training_data_args)
scaler = compute_normalization_factor(gen_training_data_args, all_data)
print("Starting train data generation")
generate_train_data(gen_training_data_args, all_data)

s3.download_file('table-linker-datasets', min_max_scaler_path , '/tmp/min_max_scaler_path.pkl')
s3.download_file('table-linker-datasets', pos_output , '/tmp/pos_features.pkl')
s3.download_file('table-linker-datasets', neg_output , '/tmp/neg_features.pkl')

print("starting training")
training_args = Namespace(num_epochs=10, lr=learning_rate, positive_feat_path='/tmp/pos_features.pkl',
                              negative_feat_path='/tmp/neg_features.pkl',
                              dev_path=dev_files, dev_output='./tmp/dev_predictions/',
                              model_save_path=model_save_path, min_max_scaler_path='/tmp/min_max_scaler_path.pkl')
best_model_path = train(training_args)
print(best_model_path)
