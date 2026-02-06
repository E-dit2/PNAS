# 设置好随机种子数
# 模型框架:esm-linear-convolution attention-res-cnn-prompt learning-pu learning
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
from statistics import mean
from torchnet import meter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef, auc, roc_curve, precision_recall_curve
from pytorch_metric_learning import losses
from sklearn.model_selection import KFold
from random import sample
import warnings
import os

warnings.filterwarnings("ignore")
import argparse

from get_model import *
import get_data
from valid_metrices import *
from unbal_loss import OversampledPULossFunc, OversampledNNPULossFunc,PUConLoss
from unbal_loss import Auc_loss, PUF1, PUPrecision, PURecall
import sys
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

type = sys.getfilesystemencoding()
sys.stdout = Logger("w.txt")

def get_train_config():
    parse = argparse.ArgumentParser(description='train model')
    parse.add_argument('-test_model', type=bool, default=False, help='是否测试模型是否可以正常运行')
    # 输入数据特征
    parse.add_argument('-msa', type=str, default='esm', help='input type:esm,evo,with evo')
    # 模型结构：是否用提示学习/pu学习/结构信息
    parse.add_argument('-prompt_learning', type=bool, default=True, help='prompt learning')
    parse.add_argument('-pu', type=bool, default=True, help='是否使用PU学习,False时,P_data={},prior=None')
    parse.add_argument('-cl', type=bool, default=True, help='是否使用对比学习')
    parse.add_argument('-structure', type=bool, default=False, help='是否使用蛋白质结构数据')
    # test set

    parse.add_argument('-nt', type=str, default='dna', help='nucleic acid type:dna/rna')
    # 参数Epoch/batch size/学习率 lr/对比损失函数的温度参数
    parse.add_argument('-epoch', type=int, default=20, help='number of iteration')  # 10
    parse.add_argument('-batch_size', type=int, default=4, help='number of samples in a batch')
    parse.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parse.add_argument('-temperature', type=float, default=0.7
                       , help='Adjust the loss')
    parse.add_argument('-a', type=float, default=0.8
                       , help='pu学习中p/P的值')
    parse.add_argument('-scheduler', type=bool, default=False, help='Adjust the learning rate of the optimizer')
    parse.add_argument('-max_metric', type=str, default='MCC',
                       help='Adjust the scheduler of the metric:AUC MCC F1 mean')
    parse.add_argument('-kfold', type=bool, default=False, help='是否进行5折交叉验证')
    # save model
    parse.add_argument('-minloss_pth', type=str, default='ml_pu_loss.pth', help='number of samples in a batch')
    parse.add_argument('-maxmetric_pth', type=str, default='ml_pu_mcc.pth', help='number of samples in a batch')
    config = parse.parse_args()
    return config


cfg = get_train_config()
if cfg.prompt_learning == True:
    if cfg.msa == 'esm':
        input_dim = 1300
        input_dim2 = 1
    elif cfg.msa == 'evo':
        input_dim = 74
        input_dim2 = 1
    else:
        input_dim = 1300
        input_dim2 = 54
else:
    if cfg.msa == 'esm':
        input_dim = 1280
        input_dim2 = 1
    elif cfg.msa == 'evo':
        input_dim = 54
        input_dim2 = 1
    else:
        input_dim = 1280
        input_dim2 = 54

if cfg.nt == 'dna':
    data_path = 'C:/Users/Administrator/Desktop/wangc/pdc_model/data/PDNA/'
    model_path = 'C:/Users/Administrator/Desktop/wangc/pdc_model/s2/model/PDNA/'
    train_files = f'{data_path}/DNA_Train_573.fa'
    test_files = f'{data_path}/DNA_Test_129.fa'
    pdb_files = f'{data_path}/pdb/'
    label_train, fasta_train = get_data.org_data(train_files, pdb_files)
    label_test, fasta_test = get_data.org_data(test_files, pdb_files)
elif cfg.nt == 'rna':
    data_path = 'C:/Users/Administrator/Desktop/wangc/pdc_model/data/PRNA/'
    model_path = 'C:/Users/Administrator/Desktop/wangc/pdc_model/s2/model/PRNA/'
    train_files = f'{data_path}/RNA-495_Train.txt'
    test_files = f'{data_path}/RNA-117_Test.txt'
    pdb_files = f'{data_path}/pdb/'
    esm_path = f'{data_path}/esm/'
    label_train, fasta_train = get_data.org_data(train_files, pdb_files)
    label_test, fasta_test = get_data.org_data(test_files, pdb_files)
    if cfg.msa != 'esm':
        eft=['5g2x_C', '5wqe_A', '5o7h_D', '5o7h_F', '5osg_hh', '6exn_aa', '6exn_oo',
         '6exn_yy', '6f3h_B', '6c0f_oo', '6d12_B', '6ff4_tt', '6ff4_uu', '6dzp_gg']
        for ID in eft:
            del label_test[ID], fasta_test[ID]
        ef = ['4kzx_aa', '4kzy_cc', '4zt0_A', '4qik_A', '1mzp_A', '3j0q_kk', '2anr_A',
              '3j81_ii', '3j81_mm', '4kzz_bb', '4pjo_ff', '4kzy_dd', '3w3s_A', '3j45_yy',
              '1ytu_B', '3amt_A', '3ice_C', '2azx_B', '3j7y_ss', '4csu_9', '4kzx_gg',
              '3b0v_D', '3jb9_cc', '2ykr_W', '3j7y_cc', '3j81_jj', '3u2e_A', '4afy_B',
              '3add_A', '2i82_A', '4rmo_O', '3j7y_bb', '3jb9_gg', '3j7y_rr', '3iab_A',
              '2gje_D', '3jb9_ee', '2gje_A', '1rpu_A', '3j7y_ee', '3j7y_gg', '4yhw_B',
              '1h2c_A', '3iab_B', '4erd_B', '3j46_nn', '3j7y_ii', '3j7y_oo', '3jb9_hh',
              '4pjo_qq', '4uyk_B', '3j7y_jj', '4uyk_A', '4kzy_ff', '2py9_C', '4kzx_ee', '2zi0_A', '3j7y_aa']
        for ID in ef:
            del label_train[ID], fasta_train[ID]
    else:
        del label_test['5g2x_C'], fasta_test['5g2x_C']

def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice


def get_list(input_list, a):
    mid_np = np.array(input_list)  # 列表转数组
    mid_np_2f = np.round(mid_np, a)  # 对数组中的元素保留a位小数
    list_new = list(mid_np_2f)
    return list_new


class text_dataset_tr(Dataset):  # 需要继承Dataset类，用于训练集
    def __init__(self, prompt_learning, ids, labels, fastas, data_path, structure, msa):
        self.prompt_learning = prompt_learning
        self.ids = ids
        self.labels = labels
        self.fastas = fastas
        self.path = data_path
        self.structure = structure
        self.msa = msa

    def __len__(self):
        if len(self.ids) == len(self.labels) == len(self.fastas):
            return len(self.ids)
        else:
            print('ERROR: dataset error!')
            raise ValueError

    def __getitem__(self, idx):
        label = self.labels[idx]
        id = self.ids[idx]
        if self.msa == 'esm':
            imput1 = get_data.get_esm(self.ids[idx], self.path)
            imput2 = get_data.get_maskesm(self.ids[idx]+'_3', self.path)
            node_features_2 = 0
        elif self.msa == 'with evo':
            imput1 = get_data.get_esm(self.ids[idx], self.path)
            node_features_2, _, _ = get_data.prepare_features(self.path, self.ids[idx], 'evo')
        else:
            imput1, _, _ = get_data.prepare_features(self.path, self.ids[idx], self.msa)
            node_features_2 = 0
        if self.prompt_learning == True:
            one_hot = get_data.get_onehot(self.fastas[idx]).unsqueeze(0)
            imput1 = torch.cat((one_hot, imput1), 2)
            imput2 = torch.cat((one_hot, imput2), 2)
        if self.structure == True:
            adj = np.load(self.path + f'dismap_files/{self.ids[idx]}_dismap.npy')
            dismap = torch.tensor(adj, dtype=torch.float).unsqueeze(0)
            return (id, imput1,imput2, node_features_2, dismap, label)
        else:
            return (id, imput1,imput2, node_features_2, None, label)

class text_dataset_te(Dataset):  # 需要继承Dataset类,用于验证集和测试集
    def __init__(self, prompt_learning, ids, labels, fastas, data_path, structure, msa):
        self.prompt_learning = prompt_learning
        self.ids = ids
        self.labels = labels
        self.fastas = fastas
        self.path = data_path
        self.structure = structure
        self.msa = msa

    def __len__(self):
        if len(self.ids) == len(self.labels) == len(self.fastas):
            return len(self.ids)
        else:
            print('ERROR: dataset error!')
            raise ValueError

    def __getitem__(self, idx):
        label = self.labels[idx]
        id = self.ids[idx]
        if self.msa == 'esm':
            node_features = get_data.get_esm(self.ids[idx], self.path)
            node_features_2 = 0
        elif self.msa == 'with evo':
            node_features = get_data.get_esm(self.ids[idx], self.path)
            node_features_2, _, _ = get_data.prepare_features(self.path, self.ids[idx], 'evo')
        else:
            node_features, _, _ = get_data.prepare_features(self.path, self.ids[idx], self.msa)
            node_features_2 = 0
        if self.prompt_learning == True:
            one_hot = get_data.get_onehot(self.fastas[idx]).unsqueeze(0)
            node_features = torch.cat((one_hot, node_features), 2)
        if self.structure == True:
            adj = np.load(self.path + f'dismap_files/{self.ids[idx]}_dismap.npy')
            dismap = torch.tensor(adj, dtype=torch.float).unsqueeze(0)
            return (id, node_features, node_features_2, dismap, label)
        else:
            return (id, node_features, node_features_2, None, label)

def get_KFolddata(ids):
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    train_ids = []
    valid_ids = []
    for train_index, test_index in kf.split(ids):
        train_ids.append([ids[i] for i in train_index])
        valid_ids.append([ids[index] for index in test_index])
    return train_ids, valid_ids


def train(cfg, net, model_path, train_data, valid_data, test_data, prior, train_pdata_len, **train_pdata):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device=device, dtype=torch.float)
    epochs = cfg.epoch
    batch_size = cfg.batch_size
    best_thre = 0
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-8)  # 使用Adam优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10,min_lr=1e-6)
    criterion1=PUConLoss()
    criterion3 = losses.SupConLoss(temperature=cfg.temperature)  # 表征结果用于对比学习
    criterion2 = nn.CrossEntropyLoss()  # 结果用的交叉熵损失函数
    criterion_upu = OversampledPULossFunc()
    criterion_nn = OversampledNNPULossFunc()
    # Begin training
    MinTrainLoss = 999
    torch.cuda.empty_cache()
    max_metric_val = 0
    m_epoch = 0
    prior_prime=0.5
    maxtrainres=[]
    maxtestres = []
    if cfg.pu == True:
        p_train = dict_slice(train_pdata, 0, train_pdata_len[0])
        P_test = dict_slice(train_pdata, train_pdata_len[0], train_pdata_len[1])
        P_valid = dict_slice(train_pdata, train_pdata_len[1], train_pdata_len[2])
    else:
        p_train = {}
        P_test = {}
        P_valid = {}

    epoch_loss1=[]
    epoch_loss2 = []
    epoch_loss=[]
    epoch_MCC=[]
    epoch_AUC=[]
    valid_iter = DataLoader(valid_data, batch_size=2, shuffle=True, collate_fn=lambda x: x)
    train_iter = DataLoader(train_data, batch_size, shuffle=True, collate_fn=lambda x: x)
    test_iter = DataLoader(test_data, batch_size=2, shuffle=True, collate_fn=lambda x: x)

    for epoch in range(1, epochs + 1):
        allbatch_train_loss = []
        allbatch_train_loss1 = []
        allbatch_train_loss2 = []
        start = time.time()
        # random.shuffle(ls)
        net.train()
        nstep = len(train_iter)
        for ii, data in enumerate(train_iter):
            train_outputs_positive = torch.empty([1,2]).cuda()
            train_outputs_unlabeled = torch.empty([1,2]).cuda()
            for (id, imput1, node_features_2, dismap, label) in data:
                imput1 = torch.tensor(imput1).to(device=device, dtype=torch.float)  # .squeeze(0)
                #imput2 = torch.tensor(imput2).to(device=device, dtype=torch.float)
                if node_features_2 != None:
                    node_features_2 = torch.tensor(node_features_2).to(device=device, dtype=torch.float)  # .squeeze(0)
                if dismap != None:
                    dismap = torch.tensor(dismap).to(device=device, dtype=torch.float)  # .squeeze(0)
                if cfg.pu == True and id in p_train.keys():
                    pu_index = p_train[id]
                    n_p = len(pu_index)
                    pu_index = torch.tensor(pu_index).to(device=device, dtype=torch.int)
                else:
                    n_p = 0
                    pu_index = None
                label0 = torch.tensor(label).to(device=device, dtype=torch.long)
                #y1_1 = torch.empty([1, 1024]).to(device=device)
                y1_2 = torch.empty([1, 2]).to(device=device)
                #y2_1 = torch.empty([1, 1024]).to(device=device)
                #y2_2 = torch.empty([1, 2]).to(device=device)
                label = torch.empty([1]).to(device=device)
                optimizer.zero_grad()  # 每一次算loss前需要将之前的梯度清零，这样才不会影响后面的更新
                output1_1,output1_2,output1_3 = net(imput1, node_features_2, dismap, pu_index)
                #output2_1,output2_2,output2_3 = net(imput2, node_features_2, dismap, pu_index)
                #y1_1 = torch.cat((y1_1, output1_1), 0)
                y1_2 = torch.cat((y1_2, output1_2), 0)
                #y2_1 = torch.cat((y2_1, output2_1), 0)
                #y2_2 = torch.cat((y2_2, output2_2), 0)
                if n_p != 0:
                    train_outputs_positive = torch.cat((train_outputs_positive, output1_3[:n_p,:]))#,output2_3[:n_p,:]))
                    train_outputs_unlabeled = torch.cat((train_outputs_unlabeled, output1_3[n_p:,:]))#,output2_3[n_p:,:]))
                #y2 = torch.empty([1, 2]).to(device=device)
                #y2 = torch.cat((y2, output2), 0)
                label = torch.cat((label, label0), 0)
            optimizer.zero_grad()
            label=label[1:]
            loss = criterion2(y1_2[1:], label.long()) #+ criterion2(y2_2[1:], label.long())
            if cfg.pu == True:
                loss_nn = criterion_nn(train_outputs_positive[1:, :], train_outputs_unlabeled[1:, :], prior,
                                       prior_prime)
                loss_upu = criterion_upu(train_outputs_positive[1:, :], train_outputs_unlabeled[1:, :], prior,
                                         prior_prime)
                if cfg.cl == True:
                    loss1 = 0#criterion3(y1_1[1:], label)# + criterion3(y2_1[1:], label))/2
                    loss2 = criterion3(y1_2[1:], label)# + criterion3(y2_2[1:], label))/2
                    #loss1 = criterion1(y1_1[1:], y2_1[1:], label)
                    #loss2 = criterion1(y1_2[1:], y2_2[1:], label)
                else:
                    loss1 = 0
                    loss2 = 0
                if loss_nn > 0:
                    loss3 = loss_upu
                else:
                    loss3 = - loss_nn
                loss = loss + loss1 + loss2 + loss3
            else:
                if cfg.cl == True:
                    loss1 =0# criterion3(y1_1[1:], label)# + criterion3(y2_1[1:], label))/2
                    loss2 = criterion3(y1_2[1:], label)# + criterion3(y2_2[1:], label))/2
                else:
                    loss1 = 0
                    loss2 = 0
                loss = loss + loss1 + loss2
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            allbatch_train_loss1.append(loss2.item())
            allbatch_train_loss2.append(loss2.item())
            allbatch_train_loss.append(loss.item())

            if ii % (nstep - 1) == 0 and ii != 0:
                nowtime = time.time()

                print('Epoch{} step{} | lr={:.6f} | train_loss={:.5f}'.format(epoch, ii,
                                                                              optimizer.param_groups[0]['lr'],
                                                                              mean(allbatch_train_loss)))
                val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc,val_prAUC = val(device, net, valid_iter, 'valid',
                                                                                  prior, val_th=None, **P_valid)
                test_th, test_rec, test_pre, test_F1, test_spe, test_mcc, test_auc,test_prAUC = val(device, net, test_iter, 'test',
                                                                                         prior, val_th, **P_test)
                if cfg.max_metric == 'AUC':
                    metrice_val = val_auc
                elif cfg.max_metric == 'MCC':
                    metrice_val = val_mcc
                elif cfg.max_metric == 'F1':
                    metrice_val = val_F1
                elif cfg.max_metric == 'PRAUC':
                    metrice_val = val_prAUC
                else:
                    print('ERROR: opt.max_metric.')
                    raise ValueError

                if metrice_val > max_metric_val:
                    max_metric_val = metrice_val
                    best_thre = val_th
                    m_epoch = epoch
                    maxtrainres=[val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc,val_prAUC]
                    maxtestres = [test_th, test_rec, test_pre, test_F1, test_spe, test_mcc, test_auc,test_prAUC]
                    save_name1 = '{}/{}'.format(model_path, cfg.maxmetric_pth)
                    torch.save([net.state_dict(), val_th, epoch], save_name1)
                if mean(allbatch_train_loss) < MinTrainLoss:
                    MinTrainLoss = mean(allbatch_train_loss)
                    save_name2 = '{}/{}'.format(model_path, cfg.minloss_pth)
                    torch.save([net.state_dict(), val_th, epoch], save_name2)  # 保存loss最小的模型
                if cfg.scheduler == True:
                    scheduler.step(metrice_val)
                epoch_loss1.append(mean(allbatch_train_loss1))
                epoch_loss2.append(mean(allbatch_train_loss2))
                epoch_loss.append(mean(allbatch_train_loss))
                epoch_MCC.append(test_mcc)
                epoch_AUC.append(test_auc)
                allbatch_train_loss.clear()
    f=open('epoch.txt','w')
    f.write('epoch_loss1:'+'\n')
    f.write(str(epoch_loss1)+'\n')
    f.write('epoch_loss2:' + '\n')
    f.write(str(epoch_loss2)+'\n')
    f.write('epoch_loss:' + '\n')
    f.write(str(epoch_loss)+'\n')
    f.write('epoch_MCC:' + '\n')
    f.write(str(epoch_MCC) + '\n')
    f.write('epoch_AUC:' + '\n')
    f.write(str(epoch_AUC) + '\n')
    f.close()
    print('{} best value:{:.3f};threshold:{} in epoch:{}'.format(cfg.max_metric, max_metric_val, best_thre, m_epoch))
    return epoch_loss1,epoch_loss2,epoch_loss,maxtrainres,maxtestres


def val(device, model, valid_iter, dataset_type, prior, val_th=None, **P_data):
    criterion_precision = PUPrecision()
    # For recall
    criterion_recall = PURecall()
    # For F1
    criterion_F1 = PUF1()
    # For Auc
    criterion_Auc = Auc_loss()
    test_precions = []
    test_recalls = []
    test_F1s = []
    auc_loss = []
    PRAUCs=[]
    model.eval()
    if val_th is not None:
        AUC_meter = meter.AUCMeter()
        Confusion_meter = meter.ConfusionMeter(k=2)
        with torch.no_grad():
            for ii, data in enumerate(valid_iter):
                test_outputs_positive = torch.empty([1,2]).cuda()
                test_outputs_negative = torch.empty([1,2]).cuda()
                positive_labels = torch.empty([1])#.cuda()
                negative_labels = torch.empty([1])
                for (id, imput_feature, node_features_2, dismap, label) in data:
                    imput_feature = imput_feature.to(device=device, dtype=torch.float)  # .squeeze(0)
                    if dismap != None:
                        dismap = torch.tensor(dismap).to(device=device, dtype=torch.float)  # .squeeze(0)
                    if node_features_2 != None:
                        node_features_2 = torch.tensor(node_features_2).to(device=device,
                                                                           dtype=torch.float)  # .squeeze(0)
                    if id in P_data.keys():
                        pu_index = P_data[id]
                        n_p = len(pu_index)
                        pu_index = torch.tensor(pu_index).to(device=device, dtype=torch.int)
                    else:
                        n_p = 0
                        pu_index = None
                    target = torch.tensor(label, dtype=int)
                    output3,output1, output2 = model(imput_feature,node_features_2, dismap, pu_index)
                    if n_p != 0:
                        test_outputs_positive = torch.cat((test_outputs_positive, output2[:n_p,:]))
                        test_outputs_negative = torch.cat((test_outputs_negative, output2[n_p:,:]))
                        positive_labels = torch.cat((positive_labels, torch.ones((n_p), dtype=torch.int32)), 0)
                        negative_labels = torch.cat(
                            (negative_labels, -torch.ones(((len(label) - n_p)), dtype=torch.int32)), 0)
                    score = output1[:, 1]
                    AUC_meter.add(score, target)
                    precision, recall, _ = precision_recall_curve(target, score.cpu())
                    PRAUCs.append(auc(recall, precision))
                    pred_bi = target.data.new(score.shape).fill_(0)
                    pred_bi[score > val_th] = 1
                    Confusion_meter.add(pred_bi, target)
                if prior != None:
                    test_output = torch.cat((test_outputs_positive[1:,1], test_outputs_negative[1:,1]), 0)
                    test_target = torch.cat((positive_labels[1:], negative_labels[1:]), 0)
                    test_precions.append(criterion_precision(test_outputs_positive[1:,1], test_outputs_negative[1:,1], prior))
                    test_recalls.append(criterion_recall(test_outputs_positive[1:,1], test_outputs_negative[1:,1], prior))
                    test_F1s.append(criterion_F1(test_outputs_positive[1:,1], test_outputs_negative[1:,1], prior))
                    auc_loss.append(criterion_Auc(test_output, test_target).item())
        test_pre = (np.mean(test_precions))
        test_rec = (np.mean(test_recalls))
        test_F1 = (np.mean(test_F1s))
        Auc = (np.mean(auc_loss))
        PRAUC=(np.mean(PRAUCs))
        val_auc = AUC_meter.value()[0]
        cfm = Confusion_meter.value()
        val_rec, val_pre, val_F1, val_spe, val_mcc = CFM_eval_metrics(cfm)
        print('{} result: '
              'th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} PRAUC={:.3f} pu_pre={:.3f} pu_rec={:.3f} pu_f1={:.3f} pu_auc={:.3f}'
              .format(dataset_type, val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc,PRAUC, test_pre, test_rec,
                      test_F1, Auc))
    else:
        AUC_meter = meter.AUCMeter()
        for j in range(2, 100, 2):
            th = j / 100.0
            locals()['Confusion_meter_' + str(th)] = meter.ConfusionMeter(k=2)
        with torch.no_grad():
            for ii, data in enumerate(valid_iter):
                test_outputs_positive = torch.empty([1,2]).cuda()
                test_outputs_negative = torch.empty([1,2]).cuda()
                positive_labels = torch.empty([1])#.cuda()
                negative_labels = torch.empty([1])#.cuda()
                for (id, imput_feature, node_features_2, dismap, label) in data:
                    imput_feature = imput_feature.to(device=device, dtype=torch.float)  # .squeeze(0)
                    if dismap != None:
                        dismap = torch.tensor(dismap).to(device=device, dtype=torch.float)  # .squeeze(0)
                    if node_features_2 != None:
                        node_features_2 = torch.tensor(node_features_2).to(device=device,
                                                                           dtype=torch.float)  # .squeeze(0)
                    if id in P_data.keys():
                        pu_index = P_data[id]
                        n_p = len(pu_index)
                        pu_index = torch.tensor(pu_index).to(device=device, dtype=torch.int)
                    else:
                        n_p = 0
                        pu_index = None
                    target = torch.tensor(label, dtype=int)
                    output3,output1, output2 = model(imput_feature,node_features_2, dismap, pu_index)
                    if n_p != 0:
                        test_outputs_positive = torch.cat((test_outputs_positive, output2[:n_p,:]))
                        test_outputs_negative = torch.cat((test_outputs_negative, output2[n_p:,:]))
                        positive_labels = torch.cat((positive_labels, torch.ones((n_p), dtype=torch.int32)), 0)
                        negative_labels = torch.cat(
                            (negative_labels, -torch.ones(((len(label) - n_p)), dtype=torch.int32)), 0)
                    score = output1[:, 1]
                    AUC_meter.add(score, target)
                    precision, recall, _ = precision_recall_curve(target, score.cpu())
                    PRAUCs.append(auc(recall, precision))
                    for j in range(2, 100, 2):
                        th = j / 100.0
                        pred_bi = target.data.new(score.shape).fill_(0)
                        pred_bi[score > th] = 1
                        locals()['Confusion_meter_' + str(th)].add(pred_bi, target)
                if prior != None:
                    test_output = torch.cat((test_outputs_positive[1:,1], test_outputs_negative[1:,1]), 0)
                    test_target = torch.cat((positive_labels[1:], negative_labels[1:]), 0)
                    test_precions.append(
                        criterion_precision(test_outputs_positive[1:,1], test_outputs_negative[1:,1], prior))
                    test_recalls.append(criterion_recall(test_outputs_positive[1:,1], test_outputs_negative[1:,1], prior))
                    test_F1s.append(criterion_F1(test_outputs_positive[1:,1], test_outputs_negative[1:,1], prior))
                    auc_loss.append(criterion_Auc(test_output, test_target).item())
        test_pre = (np.mean(test_precions))
        test_rec = (np.mean(test_recalls))
        test_F1 = (np.mean(test_F1s))
        Auc = (np.mean(auc_loss))
        PRAUC=(np.mean(PRAUCs))
        val_auc = AUC_meter.value()[0]
        val_rec, val_pre, val_F1, val_spe, val_mcc = -1, -1, -1, -1, -1
        for j in range(2, 100, 2):
            th = j / 100.0
            cfm = locals()['Confusion_meter_' + str(th)].value()
            rec, pre, F1, spe, mcc = CFM_eval_metrics(cfm)
            if mcc >= val_mcc:
                val_rec, val_pre, val_F1, val_spe, val_mcc = rec, pre, F1, spe, mcc
                val_th = th
        print('{} result: '
              'th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} PRAUC={:.3f} pu_pre={:.3f} pu_rec={:.3f} pu_f1={:.3f} pu_auc={:.3f}'
              .format(dataset_type, val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc,PRAUC, test_pre, test_rec,
                      test_F1, Auc))

    return val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc,PRAUC


def test(ml_path, net, test_data):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device=device, dtype=torch.float)
    pretrained_dict = torch.load(ml_path)
    # print(type(pretrained_dict))
    net.load_state_dict(pretrained_dict[0])
    threshold = pretrained_dict[1]
    net.eval()  # 评估模式
    print('test start')
    total_train_loss = []
    labels = []
    y_score = []
    test_iter = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    for data in test_iter:
        for (id, imput_feature, node_features_2, dismap, label) in data:
            imput_feature = imput_feature.to(device=device, dtype=torch.float)  # .squeeze(0)
            if dismap != None:
                dismap = torch.tensor(dismap).to(device=device, dtype=torch.float)  # .squeeze(0)
            if node_features_2 != None:
                node_features_2 = torch.tensor(node_features_2).to(device=device, dtype=torch.float)  # .squeeze(0)
            output3,output1, output2 = net(imput_feature, node_features_2, dismap, None)
            pre = output1
            labels.extend(label)
            y_score.extend(pre[:, 1].tolist())
    y_true = labels
    y_pred = [1 if x > threshold else 0 for x in y_score]
    test_total = len(y_true)
    c = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    print('test_total:', test_total, 'Loss:', np.mean(total_train_loss))
    fpr, tpr, auc_treshold = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    PRAUC = auc(recall, precision)
    AUC = auc(fpr, tpr)
    [tn, fn, fp, tp] = c.ravel()
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (tn, fn, fp, tp))
    print('acc:%.5f, pre:%.5f, recall:%.5f, f1:%.5f, mcc:%.5f,AUC:%.5f,PRAUC:%.5f' % (acc, p, r, f1, mcc, AUC, PRAUC))
    print('mcc:%.5f,bast_threshold:%.5f' % (mcc, threshold))
    """if all([tn, fn,fp, tp]):
        SN, SP, Pre, ACC, Re, F1, MCC = get_data.calc(tn, fn, fp, tp)
        print("ACC:%5f, Pre:%5f, Re:%5f, F1:%5f, MCC:%5f,SP:%5f, SN:%5f"%(ACC,Pre, Re, F1, MCC, SP, SN))
        print("model test finish")"""
    return [acc, p, r, f1, mcc, AUC, PRAUC]


if cfg.prompt_learning == True:
    m_net = framwork(input_dim,input_dim2)
    if cfg.pu == True:
        net = Verblizer_model(m_net)
    else:
        net = Verblizer_model(m_net)
else:
    net = framwork(input_dim,input_dim2)
# loss_type=get_loss.loss_type1
if cfg.test_model == True:
    fasta_train = dict_slice(fasta_train, 0, 10)
    label_train = dict_slice(label_train, 0, 10)
    fasta_test = dict_slice(fasta_test, 0, 5)
    label_test = dict_slice(label_test, 0, 5)
if cfg.kfold == False:
    all_ids = list(fasta_train.keys())
    train_ids = sample(all_ids, round(0.8 * len(all_ids)))
    valid_ids = list(set(all_ids) - set(train_ids))
    fasta_1 = {key: fasta_train[key] for key in train_ids}
    label_1 = {key: label_train[key] for key in train_ids}
    fasta_2 = {key: fasta_train[key] for key in valid_ids}
    label_2 = {key: label_train[key] for key in valid_ids}
    fasta_3 = fasta_test
    label_3 = label_test
    if cfg.pu == True:
        P_ls, N_ls, P, N = get_data.get_PUdata(**label_1)
        P_ls_valid, N_ls_valid, P_valid, N_valid = get_data.get_PUdata(**label_2)
        P_ls_test, N_ls_test, P_test, N_test = get_data.get_PUdata(**label_3)
        a = round(cfg.a * len(P_ls))
        p = sample(P_ls, a)
        p_train = {}
        for key in label_1.keys():
            p_train[key] = [int(i[7:]) for i in p if i[0:6] == key]
        n_p = len(P_ls)
        n_n = len(N_ls)
        n_up = n_p - len(p)
        n_u = n_n + n_up
        _prior = float(n_up) / float(n_u)
        train_pdata = {**p_train, **P_test, **P_valid}
        train_pdata_len = [len(p_train.keys()), len(p_train.keys()) + len(P_test.keys()),
                           len(p_train.keys()) + len(P_test.keys()) + len(P_valid.keys())]
    else:
        train_pdata = {}
        train_pdata_len = []
        _prior = None
    dataset_train = text_dataset_te(cfg.prompt_learning, list(label_1.keys()), list(label_1.values()),
                                 list(fasta_1.values()), data_path, cfg.structure, cfg.msa)
    dataset_valid = text_dataset_te(cfg.prompt_learning, list(label_2.keys()), list(label_2.values()),
                                 list(fasta_2.values()), data_path, cfg.structure, cfg.msa)
    dataset_test = text_dataset_te(cfg.prompt_learning, list(label_3.keys()), list(label_3.values()),
                                list(fasta_3.values()), data_path, cfg.structure, cfg.msa)
    print('the model train start')
    epoch_loss1,epoch_loss2,epoch_loss,maxtrainres,maxtestres = train(cfg, net, model_path, dataset_train, dataset_valid, dataset_test, _prior, train_pdata_len,
                           **train_pdata)
    """test(f'{model_path}/{cfg.minloss_pth}', net, dataset_test)
    if maxtestfeature[-2] != 0:
        print('maxmetric test:', cfg.max_metric)
        save_name2 = f'{model_path}/{cfg.maxmetric_pth}'
        test(save_name2, net, dataset_test)"""
else:
    train_ids, valid_ids = get_KFolddata(list(fasta_train.keys()))
    testres = []
    trainres=[]
    for i, (train_id, valid_id) in enumerate(zip(train_ids, valid_ids)):
        fasta_1 = {key: fasta_train[key] for key in train_id}
        label_1 = {key: label_train[key] for key in train_id}
        fasta_2 = {key: fasta_train[key] for key in valid_id}
        label_2 = {key: label_train[key] for key in valid_id}
        fasta_3 = fasta_test
        label_3 = label_test
        if cfg.pu == True:
            P_ls, N_ls, P, N = get_data.get_PUdata(**label_1)
            P_ls_valid, N_ls_valid, P_valid, N_valid = get_data.get_PUdata(**label_2)
            P_ls_test, N_ls_test, P_test, N_test = get_data.get_PUdata(**label_3)
            a = round(cfg.a * len(P_ls))
            p = sample(P_ls, a)
            p_train = {}
            for key in label_1.keys():
                p_train[key] = [int(i[7:]) for i in p if i[0:6] == key]
            n_p = len(P_ls)
            n_n = len(N_ls)
            n_up = n_p - len(p)
            n_u = n_n + n_up
            _prior = float(n_up) / float(n_u)
            train_pdata = {**p_train, **P_test, **P_valid}
            train_pdata_len = [len(p_train.keys()), len(p_train.keys()) + len(P_test.keys()),
                               len(p_train.keys()) + len(P_test.keys()) + len(P_valid.keys())]
        else:
            train_pdata = {}
            train_pdata_len = []
            _prior = None
        dataset_train = text_dataset_te(cfg.prompt_learning, list(label_1.keys()), list(label_1.values()),
                                     list(fasta_1.values()), data_path, cfg.structure, cfg.msa)
        dataset_valid = text_dataset_te(cfg.prompt_learning, list(label_2.keys()), list(label_2.values()),
                                     list(fasta_2.values()), data_path, cfg.structure, cfg.msa)
        dataset_test = text_dataset_te(cfg.prompt_learning, list(label_3.keys()), list(label_3.values()),
                                    list(fasta_3.values()), data_path, cfg.structure, cfg.msa)
        print('the {} train start'.format(i))
        cfg.minloss_pth = f'ml{i}_loss.pth'
        cfg.maxmetric_pth = f'ml{i}_mcc.pth'
        epoch_loss1,epoch_loss2,epoch_loss,maxtrainres,maxtestres = train(cfg, net, model_path, dataset_train, dataset_valid, dataset_test, _prior,
                               train_pdata_len, **train_pdata)
        trainres.append(maxtrainres)
        testres.append(maxtestres)
    mean1 = torch.tensor(trainres, dtype=torch.float)
    mean2 = torch.tensor(testres, dtype=torch.float)
    train_mean = torch.mean(mean1, dim=0)
    test_mean = torch.mean(mean2, dim=0)
    print('train_mean: train_th,train_rec, train_pre, train_F1, train_spe, train_mcc, train_auc,train_prauc:',
          train_mean)
    print('test_mean: test_th,test_rec, test_pre, test_F1, test_spe, test_mcc, test_auc,test_prauc:', test_mean)

"""
from matplotlib import pyplot as plt

plt.subplot(1, 2, 1)
x=range(0, cfg.epoch)
plt.plot(x, epoch_loss1,color = 'black',label ='conatt_cl_loss')
plt.plot(x, epoch_loss2,color = 'c',label ='model_cl_loss')
plt.plot(x, epoch_loss,color = 'y',label ='model_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.subplot(1, 2, 2)
plt.plot(range(0, cfg.epoch), epoch_loss)
plt.xlabel('epoch')
plt.ylabel('epoch_loss')
#plt.legend()
plt.show()  # 显示绘制的图形
"""