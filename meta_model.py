import copy
import torch
import random
from rdkit import Chem
import torch.nn as nn
from samples import sample_datasets, sample_test_datasets, sample_premeta_datasets
from model import GNN, GNN_graphpred
import torch.nn.functional as F
from loader import MoleculeDataset, _load_sider_dataset, _load_tox21_dataset
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from rdkit.Chem import PandasTools
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

import warnings
warnings.filterwarnings("ignore")
# device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

seq_voc_sml = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
# seq_voc_sml = 'CNOSFSiPClBrMgNaCaFeAsAlIBVKTlYbSbSnAgPdCoSeTiZnHLiGeCuAuNiCdInMnZrCrPtHgPbcnh1234567890-=#()[]+.'
seq_dict_sml = {v:(i+1) for i,v in enumerate(seq_voc_sml)}
seq_dict_len_sml = len(seq_dict_sml)
max_seq_len_sml = 100
class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x

class Interact_attention(nn.Module):
    def __init__(self, dim, num_tasks):
        super(Interact_attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_tasks * dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class Meta_model(nn.Module):
    def __init__(self, args):
        super(Meta_model,self).__init__()

        self.dataset = args.dataset
        self.num_tasks = args.num_tasks
        self.num_train_tasks = args.num_train_tasks
        self.num_test_tasks = args.num_test_tasks
        self.n_way = args.n_way
        self.m_support = args.m_support
        self.k_query = args.k_query
        self.gnn_type = args.gnn_type

        self.emb_dim = args.emb_dim

        self.device = args.device

        self.add_similarity = args.add_similarity
        self.add_selfsupervise = args.add_selfsupervise
        self.add_masking = args.add_masking
        self.add_weight = args.add_weight
        self.interact = args.interact

        self.batch_size = args.batch_size

        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.criterion = nn.BCEWithLogitsLoss()

        self.graph_model = GNN_graphpred(args.num_layer, args.emb_dim, 1, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        self.embedding_xds = nn.Embedding(num_embeddings=65, embedding_dim=128)
        self.conv_xds_1 = nn.Conv1d(in_channels=100, out_channels=25, kernel_size=8)
        self.conv_xds_2 = nn.Conv1d(in_channels=25, out_channels=50, kernel_size=8)
        self.conv_xds_3 = nn.Conv1d(in_channels=50, out_channels=75, kernel_size=8)
        self.conv_xds_4 = nn.Conv1d(in_channels=75, out_channels=100, kernel_size=8)
        self.fc1_xds = nn.Linear(100 * 50, 300)
        self.dropout = nn.Dropout(0.2)
        self.myl = 0.1
        if not args.input_model_file == "":
            self.graph_model.from_pretrained(args.input_model_file)

        if self.add_selfsupervise:
            self.self_criterion = nn.BCEWithLogitsLoss()

        if self.add_masking:
            self.masking_criterion = nn.CrossEntropyLoss()
            self.masking_linear = nn.Linear(self.emb_dim, 119)

        if self.add_similarity:
            self.Attention = attention(self.emb_dim)

        if self.interact:
            self.softmax = nn.Softmax(dim=0)
            self.Interact_attention = Interact_attention(self.emb_dim, self.num_train_tasks)
            
        model_param_group = []
        model_param_group.append({"params": self.graph_model.gnn.parameters()})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": self.graph_model.pool.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({"params": self.graph_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})

        if self.add_masking:
            model_param_group.append({"params": self.masking_linear.parameters()})

        if self.add_similarity:
            model_param_group.append({"params": self.Attention.parameters()})
            
        if self.interact:
            model_param_group.append({"params": self.Interact_attention.parameters()})
        
        self.optimizer = optim.Adam(model_param_group, lr=args.meta_lr, weight_decay=args.decay)

        # for name, para in self.named_parameters():
        #     if para.requires_grad:
        #         print(name, para.data.shape)
        # raise TypeError

    def update_params(self, loss, update_lr):
        grads = torch.autograd.grad(loss, self.graph_model.parameters())
        return parameters_to_vector(grads), parameters_to_vector(self.graph_model.parameters()) - parameters_to_vector(grads) * update_lr

    # def mymol2vec(self, sup_miles_list):
    #     device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
    #     smiles_list = pd.DataFrame({'smiles': sup_miles_list})
    #     # print(smiles_list)
    #     smiles_list['mol'] = smiles_list['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    #     PandasTools.AddMoleculeColumnToFrame(smiles_list, 'smiles', 'mol')
    #     model = word2vec.Word2Vec.load('model_300dim.pkl')
    #     smiles_list['sentence'] = smiles_list.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
    #     smiles_list['mol2vec'] = [DfVec(x) for x in sentences2vec(smiles_list['sentence'], model, unseen='UNK')]
    #     smile_vec = np.array([x.vec for x in smiles_list['mol2vec']])
    #     smile_vec = torch.from_numpy(smile_vec)
    #     my_one_task_emb = torch.zeros(300).to(device)
    #     my_one_task_emb = torch.mean(smile_vec, 0)
    #     return my_one_task_emb

    def seq_cat_sml(self, sml):
        x = np.zeros(max_seq_len_sml)
        for i, ch in enumerate(sml[:max_seq_len_sml]):
            x[i] = seq_dict_sml[ch]
        return x

    def build_negative_edges(self, batch):
        font_list = batch.edge_index[0, ::2].tolist()
        back_list = batch.edge_index[1, ::2].tolist()
        
        all_edge = {}
        for count, front_e in enumerate(font_list):
            if front_e not in all_edge:
                all_edge[front_e] = [back_list[count]]
            else:
                all_edge[front_e].append(back_list[count])
        
        negative_edges = []
        for num in range(batch.x.size()[0]):
            if num in all_edge:
                for num_back in range(num, batch.x.size()[0]):
                    if num_back not in all_edge[num] and num != num_back:
                        negative_edges.append((num, num_back))
            else:
                for num_back in range(num, batch.x.size()[0]):
                    if num != num_back:
                        negative_edges.append((num, num_back))

        negative_edge_index = torch.tensor(np.array(random.sample(negative_edges, len(font_list))).T, dtype=torch.long)

        return negative_edge_index

    def forward(self, epoch):
        support_loaders = []
        query_loaders = []

        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        self.graph_model.train()

        # tasks_list = random.sample(range(0,self.num_train_tasks), self.batch_size)

        for task in range(self.num_train_tasks):
            dataset = MoleculeDataset("Original_datasets/" + self.dataset + "/new/" + str(task+1), dataset = self.dataset)

            support_dataset, query_dataset, support_list, query_list = sample_datasets(dataset, self.dataset, task,self.n_way, self.m_support,self.k_query)
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)
            support_loaders.append(support_loader)
            query_loaders.append(query_loader)

            # # 打开注释
            # smi_list = support_list
            # smi_filepath = "Original_datasets/" + self.dataset + "/new/" + str(task + 1) + "/raw/tox21.json"
            # smiles_list, rdkit_mol_objs_list, labels = _load_sider_dataset(smi_filepath)
            # count = 0
            # mysmiles = []
            # for i in smi_list:
            #     count = count + 1
            #     drug_tmp = self.seq_cat_sml(smiles_list[i])
            #     mysmiles.append(drug_tmp)
            #
            # mysmiles = np.array(mysmiles)
            # mysmiles = torch.from_numpy(mysmiles)
            # mysmiles = mysmiles.long().cuda()
            # embedded_xds = self.embedding_xds(mysmiles)
            # print("=====embedded_xds=====")
            # print(embedded_xds.shape)
            # conv_xds = self.conv_xds_1(embedded_xds)
            # conv_xds = torch.relu(conv_xds)
            # conv_xds = self.dbn1(conv_xds)
            # # print("=====conv_xds=====")
            # # print(conv_xds.shape)
            # conv_xds = self.conv_xds_2(conv_xds)
            # conv_xds = torch.relu(conv_xds)
            # conv_xds = self.dbn2(conv_xds)
            # # print("=====conv_xds=====")
            # # print(conv_xds.shape)
            # conv_xds = self.conv_xds_3(conv_xds)
            # conv_xds = torch.relu(conv_xds)
            # conv_xds = self.dbn3(conv_xds)
            # # print("=====conv_xds=====")
            # # print(conv_xds.shape)
            # myxds = conv_xds.view(-1, 50 * 107)
            # # print("=====myxds.view=====")
            # # print(myxds.shape)
            # myxds = self.fc1_xds(myxds)
            # myxds = torch.mean(myxds, 0)
        for k in range(0, self.update_step):
            # print(self.fi)
            old_params = parameters_to_vector(self.graph_model.parameters())

            losses_q = torch.tensor([0.0]).to(device)

            # support_params = []
            # support_grads = torch.Tensor(self.num_train_tasks, parameters_to_vector(self.graph_model.parameters()).size()[0]).to(device)

            for task in range(self.num_train_tasks):

                # print("====task1111=====")
                # print(task)

                dataset = MoleculeDataset("Original_datasets/" + self.dataset + "/new/" + str(task + 1),
                                          dataset=self.dataset)

                mysupport_dataset, myquery_dataset, mysupport_list, myquery_list = sample_datasets(dataset, self.dataset, task,
                                                                                           self.n_way, self.m_support,
                                                                                           self.k_query)
                # 打开注释
                smi_list = mysupport_list
                query_smi_list = myquery_list
                # smi_filepath = "Original_datasets/" + self.dataset + "/new/" + str(task + 1) + "/raw/tox21.json"
                # smiles_list, rdkit_mol_objs_list, labels = _load_tox21_dataset(smi_filepath)
                smi_filepath = "Original_datasets/" + self.dataset + "/new/" + str(task + 1) + "/raw/sider.json"
                smiles_list, rdkit_mol_objs_list, labels = _load_sider_dataset(smi_filepath)
                count = 0
                mysmiles = torch.zeros(300).to(device)
                final_sup_smiles = []
                for i in smi_list:
                    count = count + 1
                    drug_tmp = self.seq_cat_sml(smiles_list[i])
                    # mysmiles.append(drug_tmp)
                    mysupsmiles = np.array(drug_tmp)
                    mysupsmiles = torch.from_numpy(mysupsmiles)
                    mysupsmiles = mysupsmiles.long().cuda()
                    embedded_xds = self.embedding_xds(mysupsmiles)
                    embedded_xds = torch.unsqueeze(embedded_xds, 0)
                    conv_sup_xds = self.conv_xds_1(embedded_xds)
                    conv_sup_xds = torch.relu(conv_sup_xds)
                    conv_sup_xds = self.conv_xds_2(conv_sup_xds)
                    conv_sup_xds = torch.relu(conv_sup_xds)
                    conv_sup_xds = self.conv_xds_3(conv_sup_xds)
                    conv_sup_xds = torch.relu(conv_sup_xds)
                    conv_sup_xds = self.conv_xds_4(conv_sup_xds)
                    conv_sup_xds = torch.relu(conv_sup_xds)
                    conv_sup_xds = conv_sup_xds.view(-1, 100 * 50)
                    mysupxds = self.fc1_xds(conv_sup_xds)
                    # mysmiles.append(mysupxds)
                    mysmiles = mysmiles + mysupxds
                    if count % 5 == 0:
                        mysmiles = torch.mean(mysmiles, 0)
                        final_sup_smiles.append(mysmiles)
                        mysmiles = torch.zeros(300).to(device)
                # print("===== count========")
                # print(count)

                # mysmiles = np.array(mysmiles)
                # mysmiles = torch.from_numpy(mysmiles)
                # mysmiles = mysmiles.long().cuda()
                #
                # embedded_xds = self.embedding_xds(mysmiles)
                #
                # conv_xds = self.conv_xds_1(embedded_xds)
                # conv_xds = torch.relu(conv_xds)
                #
                # conv_xds = self.conv_xds_2(conv_xds)
                # conv_xds = torch.relu(conv_xds)
                #
                #
                # conv_xds = self.conv_xds_3(conv_xds)
                # conv_xds = torch.relu(conv_xds)
                # conv_xds = self.conv_xds_4(conv_xds)
                # conv_xds = torch.relu(conv_xds)
                #
                # myxds = conv_xds.view(-1, 100 * 50)
                #
                # myxds = self.fc1_xds(myxds)
                # # myxds = self.dropout(myxds)
                # myxds = torch.mean(myxds, 0)
                losses_s = torch.tensor([0.0]).to(device)
                if self.add_similarity or self.interact:
                    one_task_emb = torch.zeros(300).to(device)

                # print("====final_sup_smiles====")
                # print(len(final_sup_smiles))
                for step, batch in enumerate(tqdm(support_loaders[task], desc="Iteration")):

                    # print("====support_loaders  batch=====")
                    # print(batch)
                    # print("====batch.batch=====")
                    # print(batch.batch.shape)
                    # print("====final_sup_smiles[step]====")
                    # print(len(final_sup_smiles[step]))
                    # mysupsmiles = np.array(final_sup_smiles[step])
                    # mysupsmiles = torch.from_numpy(mysupsmiles)
                    # # mysupsmiles = torch.from_numpy(mysupsmiles)
                    # mysupsmiles = mysupsmiles.long().cuda()
                    mysupxds = final_sup_smiles[step]
                    # print("==========mysupxds.shape=======")
                    # print(mysupxds.shape)
                    # embedded_sup_xds = self.embedding_xds(mysupsmiles)
                    # conv_sup_xds = self.conv_xds_1(embedded_sup_xds)
                    # conv_sup_xds = torch.relu(conv_sup_xds)
                    # conv_sup_xds = self.conv_xds_2(conv_sup_xds)
                    # conv_sup_xds = torch.relu(conv_sup_xds)
                    # conv_sup_xds = self.conv_xds_3(conv_sup_xds)
                    # conv_sup_xds = torch.relu(conv_sup_xds)
                    # conv_sup_xds = self.conv_xds_4(conv_sup_xds)
                    # conv_sup_xds = torch.relu(conv_sup_xds)
                    # conv_sup_xds = conv_sup_xds.view(-1, 100 * 50)
                    # mysupxds = self.fc1_xds(conv_sup_xds)
                    # # myxds = self.dropout(myxds)
                    # mysupxds = torch.mean(mysupsmiles, 0)

                    # print("======mysupxds=====")
                    # print(mysupxds)
                    batch = batch.to(device)

                    pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, mysupxds)
                    y = batch.y.view(pred.shape).to(torch.float64)

                    loss = torch.sum(self.criterion(pred.double(), y)) /pred.size()[0]

                    if self.add_selfsupervise:
                        positive_score = torch.sum(node_emb[batch.edge_index[0, ::2]] * node_emb[batch.edge_index[1, ::2]], dim = 1)

                        negative_edge_index = self.build_negative_edges(batch)
                        # print('negative_edge_index[0]')
                        # print(negative_edge_index[0])
                        # print('node_emb')
                        # print(node_emb.shape)
                        negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]], dim = 1)

                        self_loss = torch.sum(self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(negative_score, torch.zeros_like(negative_score)))/negative_edge_index[0].size()[0]

                        loss += (self.add_weight * self_loss)

                    if self.add_masking:
                        mask_num = random.sample(range(0,node_emb.size()[0]), self.batch_size)
                        pred_emb = self.masking_linear(node_emb[mask_num])
                        loss += (self.add_weight * self.masking_criterion(pred_emb.double(), batch.x[mask_num,0]))

                    if self.add_similarity or self.interact:
                        one_task_emb = torch.div((one_task_emb + torch.mean(node_emb,0)), 2.0)
                    # 打开注释
                    # print("==========one_task_emb 1=========")
                    # print(one_task_emb)
                    # print("==========mysupxds=========")
                    # print(mysupxds)
                    one_task_emb = one_task_emb + self.myl * mysupxds
                    # print("==========one_task_emb 2=========")
                    # print(one_task_emb)
                    losses_s += loss
    
                if self.add_similarity or self.interact:
                    if task == 0:
                        tasks_emb = []
                    tasks_emb.append(one_task_emb)

                
                new_grad, new_params = self.update_params(losses_s, update_lr = self.update_lr)

                vector_to_parameters(new_params, self.graph_model.parameters())

                this_loss_q = torch.tensor([0.0]).to(device)

                myquerysmiles = torch.zeros(300).to(device)
                final_query_smiles = []
                querycount = 0
                for i in query_smi_list:
                    querycount = querycount + 1
                    drug_tmp = self.seq_cat_sml(smiles_list[i])
                    # myquerysmiles.append(drug_tmp)

                    querysmiles = np.array(drug_tmp)
                    querysmiles = torch.from_numpy(querysmiles)
                    querysmiles = querysmiles.long().cuda()
                    embedded_xds = self.embedding_xds(querysmiles)
                    embedded_xds = torch.unsqueeze(embedded_xds, 0)
                    conv_qry_xds = self.conv_xds_1(embedded_xds)
                    conv_qry_xds = torch.relu(conv_qry_xds)
                    conv_qry_xds = self.conv_xds_2(conv_qry_xds)
                    conv_qry_xds = torch.relu(conv_qry_xds)
                    conv_qry_xds = self.conv_xds_3(conv_qry_xds)
                    conv_qry_xds = torch.relu(conv_qry_xds)
                    conv_qry_xds = self.conv_xds_4(conv_qry_xds)
                    conv_qry_xds = torch.relu(conv_qry_xds)
                    conv_qry_xds = conv_qry_xds.view(-1, 100 * 50)
                    myqryxds = self.fc1_xds(conv_qry_xds)
                    myquerysmiles = myquerysmiles + myqryxds


                    if querycount < 126:
                        if querycount % 5 == 0:
                            myquerysmiles = torch.mean(myquerysmiles, 0)
                            final_query_smiles.append(myquerysmiles)
                            myquerysmiles = torch.zeros(300).to(device)
                    if querycount == 128:
                        myquerysmiles = torch.mean(myquerysmiles, 0)
                        final_query_smiles.append(myquerysmiles)
                        myquerysmiles = torch.zeros(300).to(device)
                # print("====final_query_smiles=========")
                # print(len(final_query_smiles))
                # print(final_query_smiles[0])

                for step, batch in enumerate(tqdm(query_loaders[task], desc="Iteration")):
                    # print("====final_query_smiles=========")
                    # print(final_query_smiles[step])
                    # print("====query_loaders  batch=====")
                    # print(batch)
                    # print("====batch.batch=====")
                    # print(batch.batch.shape)

                    # my_querysmiles = np.array(final_query_smiles[step])
                    # my_querysmiles = torch.from_numpy(my_querysmiles)
                    # my_querysmiles = my_querysmiles.long().cuda()
                    # embedded_query_xds = self.embedding_xds(myquerysmiles)
                    # conv_query_xds = self.conv_xds_1(embedded_query_xds)
                    # conv_query_xds = torch.relu(conv_query_xds)
                    # conv_query_xds = self.conv_xds_2(conv_query_xds)
                    # conv_query_xds = torch.relu(conv_query_xds)
                    # conv_query_xds = self.conv_xds_3(conv_query_xds)
                    # conv_query_xds = torch.relu(conv_query_xds)
                    # conv_query_xds = self.conv_xds_4(conv_query_xds)
                    # conv_query_xds = torch.relu(conv_query_xds)
                    # myqueryxds = conv_query_xds.view(-1, 100 * 50)
                    # myqueryxds = self.fc1_xds(myqueryxds)
                    # myxds = self.dropout(myxds)
                    my_querysmiles = final_query_smiles[step]
                    # my_querysmiles = torch.mean(my_querysmiles, 0)

                    batch = batch.to(device)
                    pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, my_querysmiles)
                    # node_emb = node_emb + myqueryxds

                    y = batch.y.view(pred.shape).to(torch.float64)

                    loss_q = torch.sum(self.criterion(pred.double(), y))/pred.size()[0]

                    if self.add_selfsupervise:
                        positive_score = torch.sum(node_emb[batch.edge_index[0, ::2]] * node_emb[batch.edge_index[1, ::2]], dim = 1)

                        negative_edge_index = self.build_negative_edges(batch)
                        negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]], dim = 1)

                        self_loss = torch.sum(self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(negative_score, torch.zeros_like(negative_score)))/negative_edge_index[0].size()[0]

                        loss_q += (self.add_weight * self_loss)

                    if self.add_masking:
                        mask_num = random.sample(range(0,node_emb.size()[0]), self.batch_size)
                        pred_emb = self.masking_linear(node_emb[mask_num])
                        loss += (self.add_weight * self.masking_criterion(pred_emb.double(), batch.x[mask_num,0]))

                    this_loss_q += loss_q

                if task == 0:
                    losses_q = this_loss_q
                else:
                    losses_q = torch.cat((losses_q, this_loss_q), 0)

                vector_to_parameters(old_params, self.graph_model.parameters())

            if self.add_similarity:
                for t_index, one_task_e in enumerate(tasks_emb):
                    if t_index == 0:
                        tasks_emb_new = one_task_e
                    else:
                        tasks_emb_new = torch.cat((tasks_emb_new, one_task_e), 0)
                
                tasks_emb_new = torch.reshape(tasks_emb_new, (self.num_train_tasks, self.emb_dim))

                tasks_emb_new = tasks_emb_new.detach()
                # print("=====tasks_emb_new   add_similarity==========")
                # print(tasks_emb_new)
                # print(tasks_emb_new.shape)
                #将测试样本减去训练样本的类任务平均
                # global avg_task_emb
                # avg_task_emb = torch.zeros(300).to(device)
                # avg_task_emb = torch.mean(tasks_emb_new, dim=0)
                tasks_weight = self.Attention(tasks_emb_new)
                losses_q = torch.sum(tasks_weight * losses_q)

            elif self.interact:
                for t_index, one_task_e in enumerate(tasks_emb):
                    if t_index == 0:
                        tasks_emb_new = one_task_e
                    else:
                        tasks_emb_new = torch.cat((tasks_emb_new, one_task_e), 0)

                tasks_emb_new = tasks_emb_new.detach()
                represent_emb = self.Interact_attention(tasks_emb_new)
                represent_emb = F.normalize(represent_emb, p=2, dim=0)

                tasks_emb_new = torch.reshape(tasks_emb_new, (self.num_train_tasks, self.emb_dim))
                tasks_emb_new = F.normalize(tasks_emb_new, p=2, dim=1)
                # print("=====tasks_emb_new   interact==========")
                # print(tasks_emb_new)
                # print(tasks_emb_new.shape)
                tasks_weight = torch.mm(tasks_emb_new, torch.reshape(represent_emb, (self.emb_dim, 1)))
                print(tasks_weight)
                print(self.softmax(tasks_weight))
                print(losses_q)

                # tasks_emb_new = tasks_emb_new * torch.reshape(represent_emb_m, (self.batch_size, self.emb_dim))
                
                losses_q = torch.sum(losses_q * torch.transpose(self.softmax(tasks_weight), 1, 0))
                print(losses_q)

            else:
                losses_q = torch.sum(losses_q)
            
            loss_q = losses_q / self.num_train_tasks       
            self.optimizer.zero_grad()
            loss_q.backward()
            self.optimizer.step()
        
        return []

    def test(self, support_grads):
        accs = []
        old_params = parameters_to_vector(self.graph_model.parameters())
        for task in range(self.num_test_tasks):
            dataset = MoleculeDataset("Original_datasets/" + self.dataset + "/new/" + str(self.num_tasks-task), dataset = self.dataset)
            support_dataset, query_dataset, mytest_support_list, mytest_query_list = sample_test_datasets(dataset, self.dataset, self.num_tasks-task-1, self.n_way, self.m_support, self.k_query)
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)

            device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")

            self.graph_model.eval()

            smi_list = mytest_support_list
            query_smi_list = mytest_query_list
            smi_filepath = "Original_datasets/" + self.dataset + "/new/" + str(task + 1) + "/raw/sider.json"
            #smiles_list, rdkit_mol_objs_list, labels = _load_tox21_dataset(smi_filepath)
            smiles_list, rdkit_mol_objs_list, labels = _load_sider_dataset(smi_filepath)
            # count = 0
            # mytestsmiles = []

            # print("====smi_list====")
            # print(smi_list)
            #
            # print("====query_smi_list====")
            # print(query_smi_list)
            # for i in smi_list:
            #     count = count + 1
            #     drug_tmp = self.seq_cat_sml(smiles_list[i])
            #     mytestsmiles.append(drug_tmp)

            # print("===== count========")
            # print(count)
            count = 0
            mysmiles = torch.zeros(300).to(device)
            final_sup_smiles = []
            for i in smi_list:
                count = count + 1
                drug_tmp = self.seq_cat_sml(smiles_list[i])
                # mysmiles.append(drug_tmp)
                mysupsmiles = np.array(drug_tmp)
                mysupsmiles = torch.from_numpy(mysupsmiles)
                mysupsmiles = mysupsmiles.long().cuda()
                embedded_xds = self.embedding_xds(mysupsmiles)
                embedded_xds = torch.unsqueeze(embedded_xds, 0)
                conv_sup_xds = self.conv_xds_1(embedded_xds)
                conv_sup_xds = torch.relu(conv_sup_xds)
                conv_sup_xds = self.conv_xds_2(conv_sup_xds)
                conv_sup_xds = torch.relu(conv_sup_xds)
                conv_sup_xds = self.conv_xds_3(conv_sup_xds)
                conv_sup_xds = torch.relu(conv_sup_xds)
                conv_sup_xds = self.conv_xds_4(conv_sup_xds)
                conv_sup_xds = torch.relu(conv_sup_xds)
                conv_sup_xds = conv_sup_xds.view(-1, 100 * 50)
                mysupxds = self.fc1_xds(conv_sup_xds)
                # mysmiles.append(mysupxds)
                mysmiles = mysmiles + mysupxds
                if count % 5 == 0:
                    mysmiles = torch.mean(mysmiles, 0)
                    final_sup_smiles.append(mysmiles)
                    mysmiles = torch.zeros(300).to(device)
            for k in range(0, self.update_step_test):
                # print("=======update_step_test.k========")
                # print(k)
                loss = torch.tensor([0.0]).to(device)
                for step, batch in enumerate(tqdm(support_loader, desc="Iteration")):
                    # print("=======support_loader step========")
                    # print(step)
                    # print("=======support_loader batch========")
                    # print(batch)
                    # print("=======support_loader batch.batch========")
                    # print(batch.batch.shape)
                    # mysupsmiles = np.array(final_sup_smiles[step])
                    # mysupsmiles = torch.from_numpy(mysupsmiles)
                    # mysupsmiles = mysupsmiles.long().cuda()
                    # embedded_sup_xds = self.embedding_xds(mysupsmiles)
                    # conv_sup_xds = self.conv_xds_1(embedded_sup_xds)
                    # conv_sup_xds = torch.relu(conv_sup_xds)
                    # conv_sup_xds = self.conv_xds_2(conv_sup_xds)
                    # conv_sup_xds = torch.relu(conv_sup_xds)
                    # conv_sup_xds = self.conv_xds_3(conv_sup_xds)
                    # conv_sup_xds = torch.relu(conv_sup_xds)
                    # conv_sup_xds = self.conv_xds_4(conv_sup_xds)
                    # conv_sup_xds = torch.relu(conv_sup_xds)
                    # conv_sup_xds = conv_sup_xds.view(-1, 100 * 50)
                    # mysupxds = self.fc1_xds(conv_sup_xds)
                    # # myxds = self.dropout(myxds)
                    # mysupxds = torch.mean(mysupsmiles, 0)
                    mysupxds = final_sup_smiles[step]
                    batch = batch.to(device)

                    pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, mysupxds)
                    # print("============node_emb.shape support_loader==========")
                    # print(node_emb.shape)
                    y = batch.y.view(pred.shape).to(torch.float64)

                    loss += torch.sum(self.criterion(pred.double(), y))/pred.size()[0]

                    if self.add_selfsupervise:
                        positive_score = torch.sum(node_emb[batch.edge_index[0, ::2]] * node_emb[batch.edge_index[1, ::2]], dim = 1)

                        negative_edge_index = self.build_negative_edges(batch)
                        negative_score = torch.sum(node_emb[negative_edge_index[0]] * node_emb[negative_edge_index[1]], dim = 1)

                        self_loss = torch.sum(self.self_criterion(positive_score, torch.ones_like(positive_score)) + self.self_criterion(negative_score, torch.zeros_like(negative_score)))/negative_edge_index[0].size()[0]

                        loss += (self.add_weight *self_loss)

                    if self.add_masking:
                        mask_num = random.sample(range(0,node_emb.size()[0]), self.batch_size)
                        pred_emb = self.masking_linear(node_emb[mask_num])
                        # print("============pred_emb.shape==========")
                        # print(pred_emb.shape)
                        loss += (self.add_weight * self.masking_criterion(pred_emb.double(), batch.x[mask_num,0]))

                    print(loss)

                new_grad, new_params = self.update_params(loss, update_lr = self.update_lr)

                # if self.add_similarity:
                #     new_params = self.update_similarity_params(new_grad, support_grads)

                vector_to_parameters(new_params, self.graph_model.parameters())
                

            y_true = []
            y_scores = []
            myquerysmiles = torch.zeros(300).to(device)
            final_query_smiles = []
            querycount = 0
            querynumber = len(query_smi_list)
            lastnumber = querynumber % 5
            intnumber = querynumber -lastnumber
            for i in query_smi_list:
                querycount = querycount + 1
                drug_tmp = self.seq_cat_sml(smiles_list[i])
                # myquerysmiles.append(drug_tmp)

                querysmiles = np.array(drug_tmp)
                querysmiles = torch.from_numpy(querysmiles)
                querysmiles = querysmiles.long().cuda()
                embedded_xds = self.embedding_xds(querysmiles)
                embedded_xds = torch.unsqueeze(embedded_xds, 0)
                conv_qry_xds = self.conv_xds_1(embedded_xds)
                conv_qry_xds = torch.relu(conv_qry_xds)
                conv_qry_xds = self.conv_xds_2(conv_qry_xds)
                conv_qry_xds = torch.relu(conv_qry_xds)
                conv_qry_xds = self.conv_xds_3(conv_qry_xds)
                conv_qry_xds = torch.relu(conv_qry_xds)
                conv_qry_xds = self.conv_xds_4(conv_qry_xds)
                conv_qry_xds = torch.relu(conv_qry_xds)
                conv_qry_xds = conv_qry_xds.view(-1, 100 * 50)
                myqryxds = self.fc1_xds(conv_qry_xds)
                myquerysmiles = myquerysmiles + myqryxds

                # querynumber = querynumber - 1
                if querycount <= intnumber:
                    if querycount % 5 == 0:
                        myquerysmiles = torch.mean(myquerysmiles, 0)
                        final_query_smiles.append(myquerysmiles)
                        myquerysmiles = torch.zeros(300).to(device)
                        # print("=====final_query_smiles step=======")
                        # print(len(final_query_smiles))
                if querycount == len(query_smi_list):
                    # print("=========last==========")
                    # print(querycount)
                    # print(len(query_smi_list))
                    myquerysmiles = torch.mean(myquerysmiles, 0)
                    final_query_smiles.append(myquerysmiles)
                    myquerysmiles = torch.zeros(300).to(device)
                    # print("=====final_query_smiles step=======")
                    # print(len(final_query_smiles))

            for step, batch in enumerate(tqdm(query_loader, desc="Iteration")):
                # print("=====query_loader step=======")
                # print(step)
                # print("======query_loader batch=========")
                # print(batch)
                # print("======query_loader batch.batch=========")
                # print(batch.batch.shape)
                # querysmiles = np.array(final_query_smiles[step])
                # querysmiles = torch.from_numpy(querysmiles)
                # querysmiles = querysmiles.long().cuda()
                # embedded_query_xds = self.embedding_xds(myquerysmiles)
                # conv_query_xds = self.conv_xds_1(embedded_query_xds)
                # conv_query_xds = torch.relu(conv_query_xds)
                # conv_query_xds = self.conv_xds_2(conv_query_xds)
                # conv_query_xds = torch.relu(conv_query_xds)
                # conv_query_xds = self.conv_xds_3(conv_query_xds)
                # conv_query_xds = torch.relu(conv_query_xds)
                # conv_query_xds = self.conv_xds_4(conv_query_xds)
                # conv_query_xds = torch.relu(conv_query_xds)
                # myqueryxds = conv_query_xds.view(-1, 100 * 50)
                # myqueryxds = self.fc1_xds(myqueryxds)
                # myxds = self.dropout(myxds)
                myqueryxds = final_query_smiles[step]
                # myqueryxds = torch.mean(querysmiles, 0)
                batch = batch.to(device)

                pred, node_emb = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, myqueryxds)
                # print("============node_emb.shape==========")
                # print(node_emb.shape)
                # node_emb = node_emb - avg_task_emb
                pred = F.sigmoid(pred)
                pred = torch.where(pred>0.5, torch.ones_like(pred), pred)
                pred = torch.where(pred<=0.5, torch.zeros_like(pred), pred)
                y_scores.append(pred)
                y_true.append(batch.y.view(pred.shape))
                

            y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy()
            y_scores = torch.cat(y_scores, dim = 0).cpu().detach().numpy()
           
            roc_list = []
            roc_list.append(roc_auc_score(y_true, y_scores))
            acc = sum(roc_list)/len(roc_list)
            accs.append(acc)

            vector_to_parameters(old_params, self.graph_model.parameters())

        return accs
        
