# 源码解析：https://blog.csdn.net/weixin_42744102/article/details/87088748
''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import random
import math
import Constants as Constants
from Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"


# 在Encoder中，mask主要是为了让那些在一个batch中长度较短的序列的padding不参与attention的计算
# 而在Decoder中，还要考虑不能发生数据泄露。

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence.
        比如说，我现在的句子长度是5，在后面注意力机制的部分，
        我们在计算出来QK转置除以根号之后，softmax之前，我们得到的形状len_input * len_input  代表每个单词对其余包含自己的单词的影响力
        所以这里我需要有一个同等大小形状的矩阵，告诉我哪个位置是PAD部分，之后在计算计算softmax之前会把这里置为无穷大；
        一定需要注意的是这里得到的矩阵形状是batch_size x len_q x len_k，我们是对k中的pad符号进行标识，并没有对k中的做标识，因为没必要
        seq_q 和 seq_k 不一定一致(我自己的理解是原文是德文，翻译成英文，而原文的德语的单词个数和英语的单词个数不一样多，所以这儿可能不一致)，
        在交互注意力，q来自解码端，k来自编码端，所以告诉模型编码这边pad符号信息就可以，解码端的pad信息在交互注意力层是没有用到的；
    '''
    # 输入是两个[8,n]的[[1.],[1.]]
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    # Constants.PAD=0,".eq"找到seq_k里=0的部分
    padding_mask = seq_k.eq(Constants.PAD)  # batch_size x 1 x len_k, one is masking
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk -->batch_size x len_q x len_k
    # print(padding_mask)  # 与padding_mask纬度一样[8,n]
    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    # 输入是[8,n]的[[1.],[1.]]
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(  # 返回一个[n * n]的上三角矩阵
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls [batch_size, tgt_len, tgt_len]
    # 返回8*n*n-->8个n*n的上三角矩阵
    return subsequent_mask


def angle_matrix(input_matrix):  # 传入相对位置矩阵
    cos_array = np.zeros((input_matrix.shape[0], input_matrix.shape[1], 1))  # [8,n,1]
    # 有两个点point(x1, y1)和point(x2, y2);
    # 那么这两个点形成的斜率的角度计算方法分别是：
    # float angle = atan2(y2 - y1, x2 - x1);
    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            x = input_matrix[i][j][0]
            y = input_matrix[i][j][1]
            angle = math.atan2(y, x)  # 由x,y计算反正切值
            angle = int(angle * 180 / math.pi)  # 由反正切值计算角度
            radian = math.radians(angle)  # 角度转弧度
            cosvalue = math.cos(radian)  # 弧度转余弦值
            cos_array[i][j][0] = cosvalue  # 将余弦值写入角度矩阵
    return cos_array


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # 张量相乘
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.n_head)
                + " -> "
                + str(self.f_in)
                + " -> "
                + str(self.f_out)
                + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end):
        graph_embeded_data = []
        for start, end in seq_start_end.data:
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(  # 初始化，继承了部分父类(transforemr)的属性
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout,
            n_gatunits, n_gatheads, alpha):
        super().__init__()

        self.traj_qk_hidden2pos = nn.Linear(2, 1)  # 为了计算mask_input
        self.traj_obs_rel_linear = nn.Linear(2, 32)  # [8,n,2]-->[8,n,32]
        self.angle_linear = nn.Linear(1, 32)
        # GAT & Transformer layers初始化
        self.gatencoder = GATEncoder(n_units=n_gatunits, n_heads=n_gatheads, dropout=dropout, alpha=alpha)
        self.EncoderLayer = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.batchnorm = nn.BatchNorm1d(4, affine=False)

    def forward(self, obs_traj_rel, seq_start_end, return_attns=False):
        enc_slf_attn_list = []
        # print("src_seq",src_seq.shape)
        # -- Prepare masks --->是padding mask,具体来说就是在较短的序列后面加0
        src_seq = self.traj_qk_hidden2pos(obs_traj_rel)  # [8,n,2]--->[8,n,1]
        mask_input = src_seq.squeeze(2)  # [8,n]
        slf_attn_mask = get_attn_key_pad_mask(seq_k=mask_input, seq_q=mask_input)  # 原为seq_k=src_seq, seq_q=src_seq
        # slf_attn_mask输出为[8,n]的false
        non_pad_mask = get_non_pad_mask(mask_input)  # 原为src_seq  输出为[8,n]的1
        # 轨迹位置编码
        voc_embedding = self.traj_obs_rel_linear(obs_traj_rel)  # [8,n,32]
        # 角度编码
        angle_embedding_ndarray = angle_matrix(obs_traj_rel)  # [8,n,1]
        angle_embedding_tensor = torch.from_numpy(angle_embedding_ndarray).float().cuda()
        angle_embedding = self.angle_linear(angle_embedding_tensor)  # [8,n,32]
        # 位移距离
        movelength = abs(voc_embedding)
        # GAT编码
        enc_output_spa = self.gatencoder(voc_embedding, seq_start_end)  # [8,n,32]
        # 归一化操作
        n_size = voc_embedding.shape[1]
        batchnorm = nn.BatchNorm1d(n_size)
        batchnorm.to(device=0)
        voc_embedding = batchnorm(voc_embedding)
        angle_embedding = batchnorm(angle_embedding)
        movelength = batchnorm(movelength)
        enc_output_spa = batchnorm(enc_output_spa)

        # Transformer Encoder
        enc_finaloutput, enc_slf_attn = self.EncoderLayer(voc_embedding, angle_embedding,
                                                          non_pad_mask=non_pad_mask,
                                                          slf_attn_mask=slf_attn_mask)
        enc_slf_attn_list += [enc_slf_attn]
        enc_finaloutput, enc_slf_attn = self.EncoderLayer(enc_finaloutput, movelength,
                                                          non_pad_mask=non_pad_mask,
                                                          slf_attn_mask=slf_attn_mask)
        enc_slf_attn_list += [enc_slf_attn]
        enc_finaloutput, enc_slf_attn = self.EncoderLayer(enc_finaloutput, enc_output_spa,
                                                          non_pad_mask=non_pad_mask,
                                                          slf_attn_mask=slf_attn_mask)  # [8,n,32]
        enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_finaloutput, enc_slf_attn_list
        return enc_finaloutput, mask_input  # 应为[8,n,32]


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()
        self.encoder_output_qk_hidden2pos = nn.Linear(32, 1)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, enc_outputlinear, mask_input, enc_output, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []
        # print(tgt_obs_traj.shape)   # [8,n,2] 把预测也改成8了
        # -- Prepare masks
        src_seq = self.encoder_output_qk_hidden2pos(enc_outputlinear)  # [8,n,1]
        decoder_mask_input = src_seq.squeeze(2)  # [8,n]
        non_pad_mask = get_non_pad_mask(decoder_mask_input)  # [8,n]的[[1.],[1.]]...
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=decoder_mask_input, seq_q=decoder_mask_input)  # false矩阵
        slf_attn_mask = slf_attn_mask_keypad.gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=mask_input, seq_q=decoder_mask_input)

        # -- Forward

        dec_output = enc_outputlinear

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output,  # 上一层解码器的输出dec_output和编码器的输出enc_output
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

        #     if return_attns:
        #         dec_slf_attn_list += [dec_slf_attn]
        #
        # if return_attns:
        #     return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,  # dk=dv=dmodel/head
            tgt_emb_prj_weight_sharing,
            emb_src_tgt_weight_sharing,
            n_gatheads, n_gatunits, alpha
    ):

        super().__init__()

        self.encoder = Encoder(  # 初始化encoder
            n_layers=n_layers, n_head=n_head,
            d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner,
            dropout=dropout,
            n_gatheads=n_gatheads, n_gatunits=n_gatunits, alpha=alpha)

        self.decoder = Decoder(  # 初始化decoder
            n_layers=n_layers, n_head=n_head,
            d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner,
            dropout=dropout)
        self.decoder_output_linear = nn.Linear(32, 2)

        # 下面是一些参数设置-->共享参数
        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.encoder.angle_linear.weight = self.decoder.tgt_obs_traj_angle_linear.weight
        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            self.encoder.traj_qk_hidden2pos.weight = self.decoder.encoder_output_qk_hidden2pos.weight
            self.encoder.traj_obs_rel_linear.weight = self.decoder.tgt_obs_traj_linear.weight

    def forward(self, obs_traj_rel, seq_start_end, tgt_obs_traj, istrain):
        enc_output, mask_input = self.encoder(obs_traj_rel, seq_start_end)
        # if istrain != 0:  # 测试
        #     enc_output_linear = self.decoder_output_linear(enc_output)  # [8,n,32]-->[8,n,2]
        #     dec_output = self.decoder(enc_output_linear, mask_input, enc_output)
        # else:  # 训练&验证
        #     dec_output = self.decoder(tgt_obs_traj, mask_input, enc_output)
        dec_output = self.decoder(enc_output, mask_input, enc_output)
        dec_output = self.decoder_output_linear(dec_output)  # [8,n,2]
        return dec_output
