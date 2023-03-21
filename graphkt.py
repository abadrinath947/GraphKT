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
from graph_net import *
from tqdm import tqdm
from torch_geometric.utils.convert import to_networkx, from_networkx

data = pd.read_csv('data/as.csv', encoding = 'latin')
tag = sys.argv[1]

num_epochs = 500
batch_size = 8
block_size = 2048
train_split = 0.9

def preprocess_data(data):
    features = ['skill_id', 'correct']
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
        X = torch.tensor(batch[:, :-1, ..., :], requires_grad=True, dtype=torch.float32).cuda()
        y = torch.tensor(batch[:, 1:, ..., [0, 1]], requires_grad=True, dtype=torch.float32).cuda()
        for i in range(X.shape[1] // block_size + 1):
            if X[:, i * block_size: (i + 1) * block_size].shape[1] > 0:
                yield [X[:, i * block_size: (i + 1) * block_size], y[:, i * block_size: (i + 1) * block_size]]

def evaluate(model, batches):
    ypred, ytrue = [], []
    for X, y in batches:
        mask = y[..., -1] != -1000
        all_skill_embd = skill_net(torch.arange(110).cuda(), skill_graph.edge_index.cuda(), skill_graph.weight.cuda().float())
        skill_embd = all_skill_embd[torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()]
        ohe = torch.eye(110).cuda()[torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()]
        X = torch.cat([X, skill_embd, ohe], dim = -1)
        corrects = model.forward(X, y[..., 0])[mask]
        y = y[..., -1].unsqueeze(-1)[mask]
        ypred.append(corrects.ravel().detach().cpu().numpy())
        ytrue.append(y.ravel().detach().cpu().numpy())
    ypred = np.concatenate(ypred)
    ytrue = np.concatenate(ytrue)
    return ypred, ytrue #roc_auc_score(ytrue, ypred)

if __name__ == '__main__': 
    data_train, data_val, skill_graph = preprocess(data)

    config = GPTConfig(vocab_size = skill_graph.number_of_nodes(), block_size = block_size, 
                       n_layer = 2, n_head = 16, n_embd = 128, input_size = 130 + 110, bkt = False)
    model = GPT(config).cuda()

    skill_graph = from_networkx(skill_graph)
    skill_net = GCN(110, 128).cuda()

    print("Total Parameters:", sum(p.numel() for p in model.parameters()))

    optimizer = optim.AdamW(itertools.chain(model.parameters(), skill_net.parameters()), lr = 1e-4)
    def train(num_epochs):
        for epoch in range(num_epochs):
            model.train()
            skill_net.train()
            batches_train = construct_batches(data_train, epoch = epoch)
            pbar = tqdm(batches_train)
            losses = []
            for X, y in pbar:
                optimizer.zero_grad()
                all_skill_embd = skill_net(torch.arange(110).cuda(), skill_graph.edge_index.cuda(), skill_graph.weight.cuda().float())
                skill_embd = all_skill_embd[torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()]
                ohe = torch.eye(110).cuda()[torch.where(X[..., 0] == -1000, 0, X[..., 0]).long()]
                output = model(torch.cat([X, skill_embd, ohe], dim = -1), skill_idx = y[..., 0].detach()).ravel()
                mask = (y[..., -1] != -1000).ravel()
                loss = F.binary_cross_entropy(output[mask], y[..., -1:].ravel()[mask])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.set_description(f"Training Loss: {np.mean(losses)}")

            if epoch % 1 == 0:
                batches_val = construct_batches(data_val, val = True)
                model.eval()
                skill_net.eval()
                ypred, ytrue = evaluate(model, batches_val)
                auc = roc_auc_score(ytrue, ypred)
                acc = (ytrue == ypred.round()).mean()
                rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
                print(f"Epoch {epoch}/{num_epochs} - [VALIDATION AUC: {auc}] - [VALIDATION ACC: {acc}] - [VALIDATION RMSE: {rmse}]")
                torch.save(model.state_dict(), f"ckpts/model-{tag}-{epoch}-{auc}-{acc}-{rmse}.pth")
    train(100)
