import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from TransformerUpdate.data.loader import data_loader
from models import Transformer
from TransformerUpdate.utils import (
    displacement_error,
    final_displacement_error,
    int_tuple,
    relative_to_abs,
    get_dset_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--dataset_name", default="I80", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=8, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=16, type=int)

parser.add_argument("--noise_dim", default=(8,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)
parser.add_argument(
    "--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma"
)
parser.add_argument(
    "--hidden-units",
    type=str,
    default="16",
    help="Hidden units in each hidden layer, splitted with comma",
)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--graph_network_out_dims",
    type=int,
    default=32,
    help="dims of every node after through GAT module",
)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)

parser.add_argument("--num_samples", default=20, type=int)

parser.add_argument("--dset_type", default="test", type=str)

parser.add_argument(
    "--resume",
    default="./model_best.pth.tar",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
# Transformer
parser.add_argument('-d_model', type=int, default=32)
parser.add_argument('-d_inner', type=int, default=128)
parser.add_argument('-d_k', type=int, default=4)
parser.add_argument('-d_v', type=int, default=4)

parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-n_layers', type=int, default=4)
parser.add_argument('-dropout', type=float, default=0.1)

parser.add_argument('-embs_share_weight', action='store_true')
parser.add_argument('-proj_share_weight', action='store_true')

parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)


def evaluate_helper(error, seq_start_end):
    # 在一个张量中(本项目中为ade/fde)该方法会按start/end进行分割，按dim0相加再取最小值，最后相加得sum_
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        # print("error:", error)
        _error = error[start:end]
        # print("_error1:", _error)
        _error = torch.sum(_error, dim=0)
        # print("_error2:", _error)
        _error = torch.min(_error)
        # print("_error3:", _error)
        sum_ += _error
        # print("sum:",sum_)
    return sum_


def get_generator(checkpoint):  # 得到模型
    n_gatunits = (
            [args.traj_lstm_hidden_size]
            + [int(x) for x in args.hidden_units.strip().split(",")]
            + [args.graph_lstm_hidden_size]
    )
    n_gatheads = [int(x) for x in args.heads.strip().split(",")]
    model = Transformer(
        d_model=args.d_model, d_inner=args.d_inner,
        n_layers=args.n_layers, n_head=args.n_head,
        d_k=args.d_k, d_v=args.d_v, dropout=args.dropout,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        emb_src_tgt_weight_sharing=args.embs_share_weight,
        n_gatheads=n_gatheads, n_gatunits=n_gatunits, alpha=args.alpha
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade, fde


def evaluate(args, loader, generator):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,       # 历史轨迹
                pred_traj_gt,   # 预测的真实轨迹
                obs_traj_rel,   # 历史轨迹相对位置
                pred_traj_gt_rel, # 预测轨迹真实相对位置
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            # pred_traj_fake_rel = torch.randn(size=(8, len(obs_traj_rel[0]), 2)).cuda()
            for _ in range(args.num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj_rel, seq_start_end, obs_traj_rel, istrain=1
                )
                # 取张量pred_traj_fake_rel的后pred_len个[]，本代码中的pred_len=8(训练时为12)
                pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len:]
                # 相对位置变绝对位置，obs_traj[-1]为start pos(相对位置指相对于start pos的移动位置)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                # 真实预测部分轨迹与模型生成的预测轨迹
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
                ade.append(ade_)
                fde.append(fde_)
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)  # 平均位移误差
        fde = sum(fde_outer) / (total_traj)  # 最终位移误差
        return ade, fde


def main(args):
    checkpoint = torch.load(args.resume)
    generator = get_generator(checkpoint)
    path = get_dset_path(args.dataset_name, args.dset_type)

    _, loader = data_loader(args, path)
    ade, fde = evaluate(args, loader, generator)
    print(
        "Dataset: {}, Pred Len: {}, ADE: {:.12f}, FDE: {:.12f}".format(
            args.dataset_name, args.pred_len, ade, fde
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
