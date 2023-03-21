import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import networkx as nx
import pickle

train_split = 0.8

def create_skill_graph(df):
    if os.path.exists('skill_graph.pickle'):
        print("Using existing skill graph...")
        return pickle.load(open('skill_graph.pickle', 'rb')), pickle.load(open('skill_dict.pickle', 'rb'))

    print("Constructing skill graph...")
    df = df[~df['skill_name'].isna()]
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
    skill_dict = dict(zip(uniques, range(len(uniques))))
    pickle.dump(skill_graph, open('skill_graph.pickle', 'wb'))
    pickle.dump(skill_dict, open('skill_dict.pickle', 'wb'))
    return skill_graph, skill_dict

def preprocess(data):
    def train_test_split(data, skill_list = None):
        np.random.seed(42)
        data = data.set_index(['user_id', 'skill_name'])
        idx = np.random.permutation(data.index.unique())
        train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
        data_train = data.loc[train_idx].reset_index()
        data_val = data.loc[test_idx].reset_index()
        return data_train, data_val

    if 'skill_name' not in data.columns:
        data.rename(columns={'skill_id': 'skill_name'}, inplace=True)
    if 'original' in data.columns:
        data = data[data['original'] == 1]

    data = data[~data['skill_name'].isna() & (data['skill_name'] != 'Special Null Skill')]
    multi_col = 'template_id' if 'template_id' in data.columns else 'Problem Name'

    data_train, data_val = train_test_split(data)
    print("Train-test split finished...")

    skill_graph, skill_dict = create_skill_graph(data_train)
    print("Imputing skills...")
    repl = skill_dict[data_train['skill_name'].value_counts().index[0]]
    for skill_name in set(data_val['skill_name'].unique()) - set(skill_dict):
        skill_dict[skill_name] = repl

    print("Replacing skills...")
    data_train['skill_id'] = data_train['skill_name'].apply(lambda s: skill_dict[s])
    data_val['skill_id'] = data_val['skill_name'].apply(lambda s: skill_dict[s])

    return data_train, data_val, skill_graph
