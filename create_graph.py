import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import pickle

num_epochs = 500
batch_size = 16
block_size = 1024
train_split = 0.9

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

df = pd.read_csv('data/as.csv', encoding = 'latin')
df, _ = train_test_split(df)
df = df[~df['skill_name'].isna()]
# df = df.set_index('user_id').sample(100).reset_index()

grouped = df.groupby('user_id')['skill_name'].agg(list)
uniques = list(df['skill_name'].unique())

skill_cooccurs = {skill_name: np.zeros(df['skill_name'].nunique()) for skill_name in uniques}

for seq in tqdm(grouped.values):
    cooccur = np.zeros(df['skill_name'].nunique())
    for s in reversed(seq):
        cooccur[uniques.index(s)] += 1
        skill_cooccurs[s] = skill_cooccurs[s] + cooccur

skill_cooccurs = {k: (v / sum(v)).round(1) for k, v in skill_cooccurs.items()}
dod = {}
for i, (skill_name, edges) in enumerate(skill_cooccurs.items()):
    dod[i] = {}
    for j, e in enumerate(edges):
        if e > 0:
            dod[i][j] = {'weight': e}

skill_graph = nx.from_dict_of_dicts(dod)
pickle.dump(skill_graph, open('skill_graph.pickle', 'wb'))
pickle.dump(uniques, open('skill_mapping.pickle', 'wb'))
