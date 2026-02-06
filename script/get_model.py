import torch
import torch.nn as nn
import math


class CNNOD(nn.Module):
    def __init__(self):
        super(CNNOD, self).__init__()
        self.conv1 = nn.Conv1d(1300, 1024, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.BatchNorm1d(1024)
        self.conv2 = nn.Conv1d(1024, 128, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 2, kernel_size=5, stride=1, padding=2)
        self.head = nn.Softmax(-1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        #x = self.conv1(x)
        #x = self.norm1(x)
        #x1 = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        #x1=x1.permute(0, 2, 1)
        return self.head(x)


class Self_Attention(nn.Module):
    def __init__(self, input_dim, output_dim, num_attention_heads=4, drop_rate=0, con=True):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(output_dim / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.con = con
        self.cnnv = nn.Conv1d(input_dim, output_dim, kernel_size=7, stride=1, padding=3)
        self.norm = nn.BatchNorm1d(output_dim)
        self.linear = nn.Linear(output_dim, output_dim)
        self.dp = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(output_dim)

    def conv1d(self, x):
        x = self.cnnv(x)
        x = self.norm(x)
        return x

    def transpose_for_scores(self, x):
        if self.con == True:
            x = x.permute(0, 2, 1)
            x = self.conv1d(x)
            # x = self.linear(x)
            x = x.permute(0, 2, 1)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attention_mask=None):
        # q: bsz, protein_len, hid=heads*hidd'
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)  # q: bsz, heads, protein_len, hid'
        v = self.transpose_for_scores(v)
        attention_scores = torch.matmul(q, k.transpose(-1,
                                                       -2))  # bsz, heads, protein_len, protein_len + bsz, 1, protein_len, protein_len
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        outputs = torch.matmul(attention_probs, v)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        outputs = self.dp(outputs)
        outputs = self.ln(outputs)
        return outputs


class Conatt_block(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, num_attention_heads=2, drop_rate=0,
                 con=True, rescnn=True):
        super(Conatt_block, self).__init__()
        self.rescnn = rescnn
        self.conatt = Self_Attention(input_dim, output_dim, num_attention_heads, drop_rate, con)
        self.conv1d = nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        x1 = self.conatt(x, x, x)
        if self.rescnn == True:
            x = x.permute(0, 2, 1)
            x2 = self.conv1d(x)
            x2 = x2.permute(0, 2, 1)
            x = x1 + x2
        else:
            x = x1
        return self.ln(x)


class framwork(nn.Module):
    def __init__(self, protein_in_dim1, protein_in_dim2):
        super(framwork, self).__init__()

        self.input_block = nn.Sequential(
            nn.LayerNorm(protein_in_dim1, elementwise_affine=True)
            , nn.Linear(protein_in_dim1, 1024)
            , nn.LeakyReLU()
        )
        self.Resblock1 = Conatt_block(protein_in_dim1, 1024, kernel_size=7, stride=1, padding=3, num_attention_heads=2,
                                      drop_rate=0, con=True, rescnn=False)
        self.Resblock2 = Conatt_block(protein_in_dim2, 1024, kernel_size=7, stride=1, padding=3, num_attention_heads=2,
                                      drop_rate=0, con=True, rescnn=False)
        self.cnn = CNNOD()
        self.Resblock3=Conatt_block(1024,128, kernel_size=5, stride=1, padding=2,num_attention_heads=2, drop_rate=0,con=True,rescnn=True)
        # self.Resblock3=Conatt_block(128,64, kernel_size=3, stride=1, padding=1,num_attention_heads=2, drop_rate=0,con=True,rescnn=True)
        # self.Resblock4=Conatt_block(64,2, kernel_size=5, stride=1, padding=2,num_attention_heads=2, drop_rate=0,con=True,rescnn=True)
        self.act = nn.ReLU()
        self.logit = nn.Linear(2, 1)
        self.head = nn.Softmax(-1)

    def forward(self, x, y, z,pu):
        #x = self.input_block(x)
        x = self.Resblock1(x)
        #x =x1+x
        if y != 0:
            y = self.Resblock2(y)
        x = x + y
        x = self.act(x)
        #x = self.Resblock3(x)
        #x = self.act(x)
        y = self.cnn(x)
        if z != None:
            x = torch.matmul(z, y)
        # x = self.Resblock2(x)
        # x = self.act(x)

        #
        # x = self.Resblock4(x)
        # x = self.act(x)
        # x = self.head(x)
        # x = self.logit(x).squeeze(-1)
        return x.squeeze(0),y.squeeze(0),y.squeeze(0)


class Verblizer_model(nn.Module):  # Verblizer_model(model,device)
    def __init__(self, model):
        super(Verblizer_model, self).__init__()
        self.mode = model
        self.fc = nn.Sequential(nn.Linear(2, 2), nn.Softmax(dim=-1))
        self.list = nn.ModuleList([self.fc for _ in range(20)])

    def recognize_res(self, x):  # 默认x为n*res_feature维
        res = x[:, :, 0:20]  # res为n*20维
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        res_i = [res[0, i, :].tolist() for i in range(0, res.shape[1])]
        res_index = [i.index(1) for i in res_i]
        res_ids = [alphabet[i] for i in res_index]
        return res_ids

    def verblizer_function(self, res_ids, x):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        function_index = [alphabet.index(res_id) for res_id in res_ids]
        y = []  # .cuda()
        for index, i in enumerate(function_index):
            xi = x[index, :].unsqueeze(0)
            f = self.list[i]
            yi = f(xi)
            y.append(yi)
        y = torch.stack(y)  # .cuda()
        return y.squeeze(1)

    def pu_train(self, x, pu_index):
        x_index = torch.arange(0, x.shape[0], 1).cuda()
        p = torch.index_select(x, 0, pu_index)#.squeeze(-1)
        n_index = torch.isin(x_index, pu_index, invert=True)
        n = x[n_index]#.squeeze(-1)
        output = torch.cat((p, n), 0)
        return output

    def forward(self, x, y, z, pu_index):  # x:输入的主要特征esm/evo，y:加入的pssm/dssp/hmm特征，z:输入的结构信息，pu_index：进行pu铉锡的数据选择
        output3,output1,_= self.mode(x,y,z,None)
        res_ids = self.recognize_res(x)
        output1 = self.verblizer_function(res_ids, output1)
        #output2 = self.fc(output1)
        if pu_index != None:
            output2 = self.pu_train(output1, pu_index)
        else:
            output2 = output1
        return output3,output1,output2