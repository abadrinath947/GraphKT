import numpy as np
import sys
import itertools
import pandas as pd
import torch
from pyBKT.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
from torch import nn
import torch.optim as optim
from transformer import *
from preprocess import *
from tqdm import tqdm

data = pd.read_csv('data/as.csv', encoding = 'latin')
tag = sys.argv[1]
num_epochs = 500
batch_size = 16
block_size = 1024
train_split = 0.9

def preprocess_data(data):
    ohe_data = ohe.transform(data[ohe_columns])
    ohe_column_names = [f'ohe{i}' for i in range(len(ohe_data[0]))]
    ohe_data = pd.DataFrame(ohe_data, index = data.index, columns = ohe_column_names)
    data = data.join(ohe_data)
    #data['response_time'] = data['ms_first_response'] / 10000
    data['skill_idx'] = np.argmax(data[ohe_column_names].to_numpy(), axis = 1)
    # features = ['correct'] * 20 + ['response_time', 'attempt_count', 'hint_count', 'first_action', 'position'] + ohe_column_names
    features = ['skill_idx', 'correct'] + ohe_column_names #+ ['response_time', 'attempt_count', 'hint_count', 'first_action', 'position']
    seqs = data.groupby(['user_id']).apply(lambda x: x[features].values.tolist())
    length = min(max(seqs.str.len()), block_size)
    seqs = seqs.apply(lambda s: s[:length] + (length - min(len(s), length)) * [[-1000] * len(features)])
    return seqs

def construct_batches(raw_data, epoch = 0, val = False):
    np.random.seed(epoch)
    user_ids = raw_data['user_id'].unique()
    for _ in range(len(user_ids) // batch_size):
        user_idx = raw_data['user_id'].sample(batch_size).unique() if not val else user_ids[_ * (batch_size // 2): (_ + 1) * (batch_size // 2)]
        filtered_data = raw_data[raw_data['user_id'].isin(user_idx)].sort_values(['user_id', 'order_id'])
        batch_preprocessed = preprocess_data(filtered_data)
        batch = np.array(batch_preprocessed.to_list())
        X = torch.tensor(batch[:, :-1, ..., 1:], requires_grad=True, dtype=torch.float32).cuda()
        y = torch.tensor(batch[:, 1:, ..., [0, 1]], requires_grad=True, dtype=torch.float32).cuda()
        for i in range(X.shape[1] // block_size + 1):
            if X[:, i * block_size: (i + 1) * block_size].shape[1] > 0:
                yield [X[:, i * block_size: (i + 1) * block_size], y[:, i * block_size: (i + 1) * block_size]]

def train_test_split(data, skill_list = None):
    np.random.seed(42)
    if skill_list is not None:
        data = data[data['skill_id'].isin(skill_list)]
    data = data.set_index(['user_id'])
    idx = np.random.permutation(data.index.unique())
    train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
    data_train = data.loc[train_idx].reset_index()
    data_val = data.loc[test_idx].reset_index()
    return data_train, data_val

def evaluate(model, batches):
    ypred, ytrue = [], []
    for X, y in batches:
        mask = y[..., -1] != -1000
        corrects = model.forward(X, y[..., 0])[mask]
        y = y[..., -1].unsqueeze(-1)[mask]
        ypred.append(corrects.ravel().detach().cpu().numpy())
        ytrue.append(y.ravel().detach().cpu().numpy())
    ypred = np.concatenate(ypred)
    ytrue = np.concatenate(ytrue)
    return ypred, ytrue #roc_auc_score(ytrue, ypred)

def bkt_benchmark(train_data, test_data, **model_type):
    model = Model()
    model.fit(data = train_data, **model_type)
    print(model.params().reset_index().shape)
    return model.evaluate(data = test_data, metric = ['auc', 'accuracy', 'rmse'])


if __name__ == '__main__': 
    """
    Equation Solving Two or Fewer Steps              24253
    Percent Of                                       22931
    Addition and Subtraction Integers                22895
    Conversion of Fraction Decimals Percents         20992
    """
    data_train, data_val, skill_graph = preprocess(data)
    print("Train-test split complete...")
    ohe = OneHotEncoder(sparse = False, handle_unknown='ignore')
    ohe_columns = ['skill_id'] #, 'first_action','sequence_id', 'template_id']
    ohe.fit(data_train[ohe_columns])
    print("OHE complete...")

    batches_train = construct_batches(data_train)
    batches_val = construct_batches(data_val)
    # ASSISTMENTS: 2, 10, 60, 125
    config = GPTConfig(vocab_size = 1 * len(ohe.get_feature_names_out()), block_size = block_size, n_layer = 2, n_head = 10, n_embd = 60, input_size = 111, bkt = False)
    model = GPT(config).cuda()
    print("Total Parameters:", sum(p.numel() for p in model.parameters()))

    optimizer = optim.AdamW(model.parameters(), lr = 1e-4)
    def train(num_epochs):
        for epoch in range(num_epochs):
            model.train()
            batches_train = construct_batches(data_train, epoch = epoch)
            pbar = tqdm(batches_train)
            losses = []
            for X, y in pbar:
                optimizer.zero_grad()
                output = model(X, skill_idx = y[..., 0].detach()).ravel()
                mask = (y[..., -1] != -1000).ravel()
                loss = F.binary_cross_entropy(output[mask], y[..., -1:].ravel()[mask])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.set_description(f"Training Loss: {np.mean(losses)}")

            if epoch % 1 == 0:
                batches_val = construct_batches(data_val, val = True)
                model.eval()
                ypred, ytrue = evaluate(model, batches_val)
                auc = roc_auc_score(ytrue, ypred)
                acc = (ytrue == ypred.round()).mean()
                rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
                print(f"Epoch {epoch}/{num_epochs} - [VALIDATION AUC: {auc}] - [VALIDATION ACC: {acc}] - [VALIDATION RMSE: {rmse}]")
                torch.save(model.state_dict(), f"ckpts/model-{tag}-{epoch}-{auc}-{acc}-{rmse}.pth")
    train(2)
    """
    bkt = []
    for _ in range(5):
        bkt.append(bkt_benchmark(data_train, data_val))
    bkt_mlfgs = []
    for _ in range(5):
        bkt_mlfgs.append(bkt_benchmark(data_train, data_val, multigs = 'opportunity', multilearn = 'opportunity', forgets = True))
    bkt_mlf = []
    for _ in range(5):
        bkt_mlf.append(bkt_benchmark(data_train, data_val, multilearn = 'opportunity', forgets = True))
    bkt_mgs = []
    for _ in range(5):
        bkt_mgs.append(bkt_benchmark(data_train, data_val, multigs = 'opportunity', forgets = True))
    batches_val = construct_batches(data_val, val = True)
    model.eval()
    ypred, ytrue = evaluate(model, batches_val)
    """
