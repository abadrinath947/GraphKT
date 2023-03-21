import numpy as np
import torch

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
