import torch
import numpy as np
from numpy import linalg as LA
from Bio.PDB import PDBParser
import os
import random


def org_data(fasta_file, pdb_files):
    with open(fasta_file, 'r') as fa:  # {id:残基分类结果}
        label_ls = {}
        fasta_dict = {}
        for line in fa:
            if line[0] == '>':
                id = line[1:-1]
                pdb_path = pdb_files + f'{id}.pdb'
            elif str(0) in line and os.path.exists(pdb_path):
                ln = line[0:-1]
                label_ls[id] = [int(i) for i in ln]
            elif os.path.exists(pdb_path):
                fasta_dict[id] = line[0:-1]
            else:
                continue
    return label_ls, fasta_dict


def get_mask_fasta(id, input_fasta):
    l = len(input_fasta)
    random.seed(15)
    a = random.sample(range(0, l), 1)
    b = random.sample(range(0, l), 3)
    fasta2 = list(input_fasta)
    fasta3 = list(input_fasta)
    for i in a:
        fasta2[i] = '_'
    for i in b:
        fasta3[i] = '_'
    fasta1 = input_fasta
    fasta2 = ''.join(fasta2)
    fasta3 = ''.join(fasta3)
    mask_fasta_dict = {id + '_1': fasta1, id + '_2': fasta2, id + '_3': fasta3}
    return mask_fasta_dict


def get_esm(id, data_path):
    esm_path = data_path + f'esm/{id}.pth'
    esm_dict = torch.load(esm_path)
    esm = esm_dict[id]
    return esm #1*n*1280

def get_maskesm(id, data_path):
    esm_path = data_path + f'mask_esm/{id}.pth'
    esm_dict = torch.load(esm_path)
    esm = esm_dict[id]
    return esm #1*n*1280

def get_esm_fa(id,fasta,label,esm_path):
    # Load ESM-2 model
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    #fasta1=''.join(fasta[i] for i in sit_list)
    data = [(id, fasta)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    token_rep = token_representations[:, 1:-1]

    return token_rep

def get_onehot(fasta):
    alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    char_dict = dict((c, i) for i, c in enumerate(alphabet))
    res_encoded = [char_dict[res] for res in fasta]
    onehot_encoded = list()
    for value in res_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    e = torch.tensor(onehot_encoded)
    return e #n*20


# 获取残基距离矩阵
def get_dis_map(id,
                data_path):  # re_feature_dict={res_id_position,res_feature_matrix},其中i表示该序列的第i个位置,i从0开始（由于用氨基酸无法检索该dict）
    parser = PDBParser()
    id_pdb_path = data_path + f'AF2_predicted_pdb/{id}.pdb'
    structure = parser.get_structure(id, id_pdb_path)
    model = structure[0]
    chain = model['A']
    proteinA = []
    for re in chain:
        atom = re['CA'].get_coord()
        proteinA.append(atom)
    proteinA = np.array(proteinA)
    raw_num = proteinA.shape[0]
    xa = proteinA.reshape(raw_num, 1, 3).repeat(raw_num, axis=1)
    xb = proteinA.reshape(1, raw_num, 3).repeat(raw_num, axis=0)
    dis_map = LA.norm(xa - xb, ord=2, axis=2)
    min_val = np.min(dis_map)
    max_val = np.max(dis_map)
    normalized_dis_map = (dis_map - min_val) / (max_val - min_val)  # 对矩阵进行归一化处理
    dis_map = np.exp(-normalized_dis_map)  # 再取e^(-x)
    # mid=(np.max(dis_map)-np.min(dis_map))/2 + np.min(dis_map)
    # print('min:',np.min(dis_map),'max',np.max(dis_map))

    return dis_map


def get_site(id, data_path):
    pdb_path = data_path + f'AF2_predicted_pdb/{id}.pdb'
    parser = PDBParser()
    structure = parser.get_structure(id, pdb_path)
    model = structure[0]
    chain = model['A']
    proteinA = []
    for re in chain:
        atom = re['CA'].get_coord()
        proteinA.append(atom)
    site = torch.tensor(proteinA)
    return site

def prepare_features(data_path, ID, msa):
    if msa == "both" or msa == "single":
        AF2_single = np.load(data_path + f'npy_files/{ID}.npy')
    if msa == "both" or msa == "evo":
        pssm = np.load(data_path + f'PSSM/{ID}_pssm.npy')
        hhm = np.load(data_path + f'HMM/{ID}_hhm.npy')
    dssp = np.load(data_path + f'dssp_files/{ID}_dssp.npy')
    if msa == "both":
        node_features = np.hstack([AF2_single, pssm, hhm, dssp])#node_dim =  384 + 40 + 14=438
    elif msa == "single":
        node_features = np.hstack([AF2_single, dssp])#node_dim =  384 + 14=398
    elif msa == "evo":
        node_features = np.hstack([pssm, hhm, dssp])#node_dim = 40 + 14=54

    dismap = np.load(data_path + f'dismap_files/{ID}_dismap.npy')

    node_features = torch.tensor(node_features, dtype = torch.float).unsqueeze(0)
    dismap = torch.tensor(dismap, dtype = torch.float).unsqueeze(0)

    masks = np.ones(node_features.shape[1])
    masks = torch.tensor(masks, dtype = torch.long).unsqueeze(0)

    return node_features, dismap, masks

def get_input(ID, data_path):
    dssp_path = data_path + 'dssp_files/'
    esm = get_esm(ID, data_path)
    dssp = np.load(dssp_path + f'{ID}_dssp.npy')  # 这里需要先构建_dssp.npy文件
    dismap = get_dis_map(ID, data_path)

    esm = esm.squeeze(0).numpy()
    dismap = torch.tensor(dismap, dtype=torch.float)
    feature = np.hstack([esm, dssp])
    return feature, dismap


def d_n(dis_map, p, m):  # p为positive的位置列表，m为距离矩阵中最大的距离
    l1 = p * len(p)
    l2 = []
    for s in p:
        l2.extend([s] * len(p))
    index = (
        torch.LongTensor(l1),
        torch.LongTensor(l2),
    )
    l3 = [m] * len(l1)

    new_value = torch.Tensor(l3)
    d = dis_map.index_put(index, new_value)
    return d


def get_PNdata(data_path, **label_ls):
    ls = list(label_ls.keys())
    RN = {}
    p = {}
    u = {}
    for id in ls[0:2]:
        l = list(label_ls[id])
        dis_map = torch.Tensor(get_dis_map(id, data_path))
        p_p = [i for i, v in enumerate(l) if v == 1]
        D = []
        m = torch.max(dis_map)
        d = d_n(dis_map, p_p, m)  # 将所有p-p设置魏最大距离
        for i in p_p:
            n = int(torch.argmin(d[:, i]))
            d[n, :] = m
            D.append(n)
        D = list(set(D))
        A = list(range(len(l)))
        RN[id] = D
        p[id] = p_p
        if len(p[id])!=(l == 1).sum:
            print('ERROR: dataset error!')
            raise ValueError
        u[id] = list(set(A) - set(D) - set(p_p))
    return p, RN, u  # data是一个{id：p&ln的index}


def get_PUdata(**train_label):
    N = {}
    P = {}
    P_ls=[]
    N_ls=[]
    for id in train_label.keys():
        l = list(train_label[id])
        AP = [i for i, v in enumerate(l) if v == 1]
        A = list(range(len(l)))
        N[id] = list(set(A) - set(AP))  
        P[id] = AP
        pa=[id+'_'+str(i) for i in AP]
        na=[id+'_'+str(i) for i in N[id]]
        P_ls.extend(pa)
        N_ls.extend(na)
    return P_ls,N_ls,P,N 


def get_dis(predict):
    p = torch.tensor([0, 1])
    n = torch.tensor([0, -1])
    d1 = torch.pairwise_distance(predict, p)
    d2 = torch.pairwise_distance(predict, n)
    return d1, d2


def calc(TN, FN, FP,TP):
    SN = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    Pre = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    Re = TP / (TP + FN)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    fz = TP * TN - FP * FN
    fm = (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
    MCC = fz / pow(fm, 0.5)
    return SN, SP, Pre, ACC, Re, F1, MCC
