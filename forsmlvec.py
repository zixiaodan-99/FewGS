import sys
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import urllib.request
import numpy as np
import pandas as pd
import io
import os
from rdkit.Chem import PandasTools
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
import collections
from loader import MoleculeDataset, _load_sider_dataset
from samples import sample_datasets, sample_test_datasets, sample_premeta_datasets

def smiles2vec(sup_miles_list):
    smiles_list =pd.DataFrame({'smiles': sup_miles_list})
        # print(smiles_list)
    smiles_list['mol']=smiles_list['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    PandasTools.AddMoleculeColumnToFrame(smiles_list,'smiles','mol')
    model = word2vec.Word2Vec.load('model_300dim.pkl')
    smiles_list['sentence'] = smiles_list.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
    smiles_list['mol2vec'] = [DfVec(x) for x in sentences2vec(smiles_list['sentence'], model, unseen='UNK')]
    smile_vec = np.array([x.vec for x in smiles_list['mol2vec']])
    smile_vec = torch.from_numpy(smile_vec)
    my_one_task_emb = torch.zeros(300).to(device)
    my_one_task_emb = torch.mean(smile_vec, 0)
    return my_one_task_emb