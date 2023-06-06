import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-dark")
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

parser.add_argument("--dataset_name", default="US101", type=str)
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

def evaluate_helper(error, seq_start_end, model_output_traj, model_output_traj_best):
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        min_index = _error.min(0)[1].item()
        model_output_traj_best[:, start:end, :] = model_output_traj[min_index][
            :, start:end, :
        ]
    return model_output_traj_best


def get_generator(checkpoint):
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


def plot_trajectory(args, loader, generator):
    ground_truth_input = []
    all_model_output_traj = []
    ground_truth_output = []
    pic_cnt = 0

    num1 = 0
    num2 = 0
    num3 = 0
    with torch.no_grad():
        for batch in loader:
            num1 += 1
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch
            ade = []
            ground_truth_input.append(obs_traj)
            ground_truth_output.append(pred_traj_gt)
            model_output_traj = []
            model_output_traj_best = torch.ones_like(pred_traj_gt).cuda()

            for _ in range(args.num_samples):  # num_samples=20
                pred_traj_fake_rel = generator(
                    obs_traj_rel, seq_start_end, obs_traj_rel, istrain=1
                )
                pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len :]

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                model_output_traj.append(pred_traj_fake)
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
                ade.append(ade_)
            model_output_traj_best = evaluate_helper(
                ade, seq_start_end, model_output_traj, model_output_traj_best
            )
            all_model_output_traj.append(model_output_traj_best)

            # print("seq_start_end:", seq_start_end)

            for (start, end) in seq_start_end:   # 例如 seq_star_end:[[0,2],[2,4]],那么迭代两次(start,end):(0,2)/(2,4)
                num2 += 1  # 956
                plt.figure(figsize=(20,15), dpi=100)
                ground_truth_input_x_piccoor = (
                    obs_traj[:, start:end, :].cpu().numpy()[:, :, 0].T
                )
                ground_truth_input_y_piccoor = (
                    obs_traj[:, start:end, :].cpu().numpy()[:, :, 1].T
                )
                ground_truth_output_x_piccoor = (
                    pred_traj_gt[:, start:end, :].cpu().numpy()[:, :, 0].T
                )
                ground_truth_output_y_piccoor = (
                    pred_traj_gt[:, start:end, :].cpu().numpy()[:, :, 1].T
                )
                model_output_x_piccoor = (
                    model_output_traj_best[:, start:end, :].cpu().numpy()[:, :, 0].T
                )
                model_output_y_piccoor = (
                    model_output_traj_best[:, start:end, :].cpu().numpy()[:, :, 1].T
                )
                # print("start,end:", (start,end))
                # # 以下参数shape均为[end-start,8],8应为预测轨迹长度
                # print("ground_truth_input_x_piccoor:", ground_truth_input_x_piccoor.shape,
                #       "ground_truth_input_y_piccoor:", ground_truth_input_y_piccoor.shape,
                #       "ground_truth_output_x_piccoor:", ground_truth_output_x_piccoor.shape,
                #       "ground_truth_output_y_piccoor:", ground_truth_output_y_piccoor.shape,
                #       "model_output_x_piccoor:", model_output_x_piccoor.shape,
                #       "model_output_y_piccoor:", model_output_y_piccoor.shape)
                for i in range(ground_truth_output_x_piccoor.shape[0]):
                    num3 += 1  # 6622
                    observed_line = plt.plot(
                        ground_truth_input_x_piccoor[i, :],
                        ground_truth_input_y_piccoor[i, :],
                        "r-",
                        linewidth=4,
                        label="Observed Trajectory",
                    )[0]
                    observed_line.axes.annotate(
                        "",
                        xytext=(
                            ground_truth_input_x_piccoor[i, -2],
                            ground_truth_input_y_piccoor[i, -2],
                        ),
                        xy=(
                            ground_truth_input_x_piccoor[i, -1],
                            ground_truth_input_y_piccoor[i, -1],
                        ),
                        arrowprops=dict(
                            arrowstyle="->", color=observed_line.get_color(), lw=1
                        ),
                        size=20,
                    )
                    ground_line = plt.plot(
                        np.append(
                            ground_truth_input_x_piccoor[i, -1],
                            ground_truth_output_x_piccoor[i, :],
                        ),
                        np.append(
                            ground_truth_input_y_piccoor[i, -1],
                            ground_truth_output_y_piccoor[i, :],
                        ),
                        "b-",
                        linewidth=4,
                        label="Ground Truth",
                    )[0]
                    predict_line = plt.plot(
                        np.append(
                            ground_truth_input_x_piccoor[i, -1],
                            model_output_x_piccoor[i, :],
                        ),
                        np.append(
                            ground_truth_input_y_piccoor[i, -1],
                            model_output_y_piccoor[i, :],
                        ),
                        color="#ffff00",
                        ls="--",
                        linewidth=4,
                        label="Predicted Trajectory",
                    )[0]

                # plt.axis("off")
                plt.savefig(
                    "./traj_fig/pic_{}.png".format(pic_cnt)
                )
                plt.close()
                pic_cnt += 1

    print("num1:", num1)
    print("num2:", num2)
    print("num3:", num3)
def main(args):
    checkpoint = torch.load(args.resume) # 模型加载函数
    generator = get_generator(checkpoint)  # 生成器
    path = get_dset_path(args.dataset_name, args.dset_type)
    # print("path:",path)  # D:\pythonproject\SST\STEGCN\datasets\zara2\test
    _, loader = data_loader(args, path)
    plot_trajectory(args, loader, generator)


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.mkdir("./traj_fig")
    main(args)
