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
from utils import *
from tqdm import tqdm
from torch_geometric.utils.convert import to_networkx, from_networkx

data = pd.read_csv('data/as.csv', encoding = 'latin')
tag = sys.argv[1]

num_epochs = 500
batch_size = 8
block_size = 2048
train_split = 0.9

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
