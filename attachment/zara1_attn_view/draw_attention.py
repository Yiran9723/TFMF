import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

# from attrdict import AttrDict

from gat.data.loader import data_loader
from gat.models import TrajectoryGenerator
from gat.losses import displacement_error, final_displacement_error
from gat.utils import relative_to_abs, get_dset_path
from gat.utils import int_tuple, bool_flag, get_total_norm
import utils
import time
from IPython import embed

torch.manual_seed(72)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

# Dataset options
parser.add_argument("--dataset_name", default="zara1", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=8, type=int)
parser.add_argument("--skip", default=1, type=int)


parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)


parser.add_argument("--noise_dim", default=(0,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

parser.add_argument("--traj_lstm_num_layers", default=1, type=int)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)


# graph attention network

parser.add_argument("--graph_lstm_num_layers", default=1, type=int)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--graph_node_output_dims",
    type=int,
    default=32,
    help="dims of every node after through a single GAT leayer",
)
parser.add_argument(
    "--nb_heads", type=int, default=1, help="Number of head attentions."
)
parser.add_argument(
    "--graph_network_out_dims",
    type=int,
    default=32,
    help="dims of every node after through GAT module( equal to lstm input size)",
)

parser.add_argument("--pred_lstm_num_layers", default=1, type=int)
parser.add_argument("--pred_lstm_input_size", default=16, type=int)

parser.add_argument(
    "--hidden_size_before_noise",
    default=64,
    type=int,
    help="dims after MLP operator on (traj_lstm_hidden_size + graph_lstm_hidden_size)",
)

parser.add_argument("--num_samples", default=1, type=int)


parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)

parser.add_argument("--dset_type", default="test", type=str)


parser.add_argument(
    "--resume",
    default="./model_best.pth.tar",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)


def get_generator(checkpoint):
    model = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        graph_node_output_dims=args.graph_node_output_dims,
        graph_network_out_dims=args.graph_network_out_dims,
        pred_lstm_input_size=args.pred_lstm_input_size,
        pred_lstm_num_layers=args.pred_lstm_num_layers,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        traj_lstm_input_size=args.traj_lstm_input_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def attention_view(args, loader, generator):
    with torch.no_grad():
        for batch in loader:
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
            pred_traj_fake_rel = generator(
                obs_traj_rel, obs_traj, seq_start_end, 1, True, 2
            )


def main(args):
    checkpoint = torch.load(args.resume)
    generator = get_generator(checkpoint)
    path = get_dset_path(args.dataset_name, args.dset_type)
    _, loader = data_loader(args, path)
    attention_view(args, loader, generator)
    i = 0
    for filename in sorted(os.listdir("./attn_fig/")):
        dst = str(i) + ".pdf"
        src = "./attn_fig/" + filename
        dst = "./attn_fig/" + dst
        os.rename(src, dst)
        i += 1


if __name__ == "__main__":
    args = parser.parse_args()
    os.mkdir("./attn_fig/")
    main(args)

