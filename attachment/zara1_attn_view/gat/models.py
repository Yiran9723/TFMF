import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from IPython import embed
import time
import random
import matplotlib.pyplot as plt


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(MultiHeadGraphAttention, self).__init__()
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
        n = h.size(0)  # h is of size n x f_in
        h_prime = torch.matmul(h.unsqueeze(0), self.w)  #  n_head x n x f_out
        attn_src = torch.bmm(h_prime, self.a_src)  # n_head x n x 1
        attn_dst = torch.bmm(h_prime, self.a_dst)  # n_head x n x 1
        attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(
            0, 2, 1
        )  # n_head x n x n

        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)  # n_head x n x n
        attn = self.dropout(attn)
        output = torch.bmm(attn, h_prime)  # n_head x n x f_out

        if self.bias is not None:
            return (output + self.bias), attn
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
    def __init__(self, n_units=[32, 16, 32], n_heads=[8, 1], dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                MultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.bn_list = [
            torch.nn.BatchNorm1d(32).cuda(),
            torch.nn.BatchNorm1d(64).cuda(),
        ]

    def forward(self, x, curr_seq_pos, attn_view):
        n = x.shape[0]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.bn_list[i](x)
            x, attn = gat_layer(x)  # n_head x n x f_out
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=0)
            else:
                x = F.elu(x.transpose(0, 1).contiguous().view(n, -1))
                x = F.dropout(x, self.dropout, training=self.training)

        if attn_view:
            attn = attn.squeeze(0)
            return x, attn
        else:
            return x


class GATEncoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""

    def __init__(self, nfeat=16, nhid=8, nclass=32, dropout=0.6, alpha=0.2):
        super(GATEncoder, self).__init__()
        self.graph_node_input_dims = nfeat
        n_units = [32, 16, 32]
        n_heads = [4, 1]
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, obs_traj_pos, seq_start_end, attn_view):
        graph_embeded_data = []
        one_time_attn = []
        for start, end in seq_start_end.data:
            curr_seq_embedding_traj = obs_traj_embedding[start:end, :]
            curr_seq_pos = obs_traj_pos[start:end, :]
            if attn_view:
                curr_seq_graph_embedding, curr_seq_attn = self.gat_net(
                    curr_seq_embedding_traj, curr_seq_pos, attn_view
                )
                one_time_attn.append(curr_seq_attn)
            else:
                curr_seq_graph_embedding = self.gat_net(
                    curr_seq_embedding_traj, curr_seq_pos, attn_view
                )
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=0)
        if attn_view:
            return graph_embeded_data, one_time_attn
        else:
            return graph_embeded_data


class TrajectoryGenerator(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        graph_node_output_dims,
        graph_network_out_dims,
        pred_lstm_input_size,
        pred_lstm_num_layers,
        dropout,
        alpha,
        graph_lstm_hidden_size,
        traj_lstm_hidden_size,
        traj_lstm_input_size,
        noise_dim=(8,),
        noise_type="gaussian",
    ):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len

        self.gatencoder = GATEncoder(
            nfeat=traj_lstm_hidden_size,
            nhid=graph_node_output_dims,
            nclass=graph_network_out_dims,
            dropout=dropout,
            alpha=alpha,
        )

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size

        self.pred_lstm_hidden_size = (
            self.traj_lstm_hidden_size + self.graph_lstm_hidden_size + noise_dim[0]
        )
        self.pred_lstm_num_layers = pred_lstm_num_layers
        self.pred_lstm_input_size = pred_lstm_input_size

        self.traj_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size)
        self.graph_lstm_model = nn.LSTMCell(
            graph_network_out_dims, graph_lstm_hidden_size
        )

        self.traj_hidden2pos = nn.Linear(self.traj_lstm_hidden_size, 2)
        self.traj_gat_hidden2pos = nn.Linear(
            self.traj_lstm_hidden_size + self.graph_lstm_hidden_size, 2
        )
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, 2)

        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.pred_lstm_model = nn.LSTMCell(
            traj_lstm_input_size, self.pred_lstm_hidden_size
        )

    def init_hidden_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_hidden_graph_lstm(self, batch):
        return (
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
        )

    def add_noise(self, _input, seq_start_end):
        noise_shape = (seq_start_end.size(0),) + self.noise_dim  # example [64,8]

        z_decoder = get_noise(noise_shape, self.noise_type)

        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end - start, 1)
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h

    def forward(
        self,
        obs_traj_rel,
        obs_traj_pos,
        seq_start_end,
        teacher_forcing_ratio=0.5,
        attn_view=False,
        training_step=4,
    ):
        batch = obs_traj_rel.shape[1]
        last_pos_rel = obs_traj_rel[-1]
        pred_traj_rel = []

        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(batch)

        all_time_attn = []
        all_time_rel_pos = []

        for i, input_t in enumerate(
            obs_traj_rel[: self.obs_len].chunk(
                obs_traj_rel[: self.obs_len].size(0), dim=0
            )
        ):
            # teacher_force = random.random() < teacher_forcing_ratio
            # if i > 1 and self.training:
            #    input_t = input_t if teacher_force else output.unsqueeze(0)
            # traj_embeded = self.traj_embedding(input_t.squeeze(0))
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )

            if training_step == 1:
                output = self.traj_hidden2pos(traj_lstm_h_t)
                pred_traj_rel += [output]

            elif training_step == 2:
                graph_lstm_input, one_time_attn = self.gatencoder(
                    traj_lstm_h_t, obs_traj_pos[i], seq_start_end, attn_view
                )
                all_time_attn.append(one_time_attn)

            elif training_step == 3:
                graph_lstm_input = self.gatencoder(
                    traj_lstm_h_t, obs_traj_pos[i], seq_start_end, attn_view
                )
                graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
                    graph_lstm_input, (graph_lstm_h_t, graph_lstm_c_t)
                )
                encoded_before_noise_hidden = torch.cat(
                    (traj_lstm_h_t.squeeze(0), graph_lstm_h_t.squeeze(0)), dim=1
                )
                output = self.traj_gat_hidden2pos(encoded_before_noise_hidden)
                pred_traj_rel += [output]

            else:
                graph_lstm_input = self.gatencoder(
                    traj_lstm_h_t, obs_traj_pos[i], seq_start_end, attn_view
                )
                graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
                    graph_lstm_input, (graph_lstm_h_t, graph_lstm_c_t)
                )

        if training_step == 1 or training_step == 3:
            return torch.stack(pred_traj_rel)
        elif training_step == 2:
            every_group_attn = []
            every_group_rel_pos = []
            attn_loss = torch.zeros(1).to(obs_traj_rel)
            for t1 in range(len(all_time_attn[0])):
                every_group_attn_tmp = []
                for t2 in range(8):
                    every_group_attn_tmp.append(all_time_attn[t2][t1])
                every_group_attn_tmp = torch.stack(every_group_attn_tmp)

                every_group_attn.append(every_group_attn_tmp)

                curr_seq_percnt = (seq_start_end[t1][1] - seq_start_end[t1][0]).item()
                if curr_seq_percnt < 4:
                    continue
                fig, ax = plt.subplots()
                plt.gca().set_aspect("equal", adjustable="box")
                ground_truth_input_x = (
                    obs_traj_pos[:, seq_start_end[t1][0] : seq_start_end[t1][1], :][
                        :, :, 0
                    ]
                    .permute(1, 0)
                    .cpu()
                    .numpy()
                )
                ground_truth_input_y = (
                    obs_traj_pos[:, seq_start_end[t1][0] : seq_start_end[t1][1], :][
                        :, :, 1
                    ]
                    .permute(1, 0)
                    .cpu()
                    .numpy()
                )

                for per in range(curr_seq_percnt):
                    observed_line = plt.plot(
                        ground_truth_input_x[per, :],
                        ground_truth_input_y[per, :],
                        "k-o",
                        markevery=[1, 3, 5, 7],
                        markersize=2,
                        linewidth=1,
                        label="person_{}".format(per),
                    )[0]
                    observed_line.axes.annotate(
                        "",
                        xytext=(
                            ground_truth_input_x[per, -5],
                            ground_truth_input_y[per, -5],
                        ),
                        xy=(
                            ground_truth_input_x[per, -4],
                            ground_truth_input_y[per, -4],
                        ),
                        arrowprops=dict(
                            arrowstyle="->", color=observed_line.get_color(), lw=1
                        ),
                        size=12,
                    )

                target_person_id = 1
                surrounding_person_id = [
                    i for i in range(curr_seq_percnt) if i != target_person_id
                ]
                time_stamp = str(datetime.now())
                for att_idx in [1, 3, 5, 7]:
                    curr_attn = (
                        every_group_attn[t1][att_idx][target_person_id].cpu().numpy()
                    )
                    all_attn_value = list(curr_attn)
                    all_attn_value.remove(curr_attn[target_person_id])
                    all_attn_value_new = [i for i in all_attn_value]
                    curr_attn_soft = softmax(all_attn_value_new)
                    atten_tmp = np.zeros_like(curr_attn)
                    atten_tmp[surrounding_person_id] = curr_attn_soft

                    color = next(ax._get_lines.prop_cycler)["color"]
                    for k in surrounding_person_id:
                        circle = plt.Circle(
                            (
                                ground_truth_input_x[k, att_idx],
                                ground_truth_input_y[k, att_idx],
                            ),
                            atten_tmp[k],
                            fill=False,
                            color=color,
                        )
                        ax.add_artist(circle)

                ax.set_xlim([0, 15])
                ax.set_ylim([0, 15])

                ax.set_xticks([])
                ax.set_yticks([])

                plt.savefig("./attn_fig/{}.pdf".format(time_stamp), bbox_inches="tight")
                plt.close()
        else:
            encoded_before_noise_hidden = torch.cat(
                (traj_lstm_h_t, graph_lstm_h_t), dim=1
            )
            pred_lstm_hidden = self.add_noise(
                encoded_before_noise_hidden, seq_start_end
            )

            pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
            output = obs_traj_rel[-1]
            if self.training:
                for i, input_t in enumerate(
                    obs_traj_rel[-self.pred_len :].chunk(
                        obs_traj_rel[-self.pred_len :].size(0), dim=0
                    )
                ):
                    teacher_force = random.random() < teacher_forcing_ratio
                    input_t = input_t if teacher_force else output.unsqueeze(0)
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [output]
                outputs = torch.stack(pred_traj_rel)
            else:
                for i in range(self.pred_len):
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        output, (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [output]
                outputs = torch.stack(pred_traj_rel)
            return outputs
