import argparse
import logging
import os
import random
import numpy as np
import shutil

import torch
import torch.nn as nn
import adabound
import torch.backends.cudnn as cudnn

import utils
from gat.data.loader import data_loader
from gat.losses import l2_loss
from gat.losses import displacement_error, final_displacement_error
from gat.models import TrajectoryGenerator
from gat.utils import int_tuple
from gat.utils import relative_to_abs, get_dset_path

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
parser.add_argument("--num_epochs", default=1000, type=int)

parser.add_argument("--noise_dim", default=(0,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

parser.add_argument("--traj_lstm_num_layers", default=1, type=int)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)


parser.add_argument("--clipping_threshold_g", default=2.0, type=float)

parser.add_argument(
    "--lr",
    default=1e-3,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)

parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)

parser.add_argument("--graph_lstm_num_layers", default=1, type=int)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--graph_node_output_dims",
    type=int,
    default=32,
    help="dims of every node after through a single GAT leayer",
)
parser.add_argument(
    "--nb_heads", type=int, default=10, help="Number of head attentions."
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

parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)

parser.add_argument("--best_k", default=1, type=int)

parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)

parser.add_argument("--print_every", default=10, type=int)

# Misc
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)

parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)


best_ade = 100


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, "train")
    val_path = get_dset_path(args.dataset_name, "test")

    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logging.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

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
    model.cuda()
    optimizer = adabound.AdaBound(
        [
            {"params": model.traj_lstm_model.parameters(), "lr": 1e-2},
            {"params": model.traj_hidden2pos.parameters()},
            {"params": model.gatencoder.parameters(), "lr": 1e-2},
            {"params": model.graph_lstm_model.parameters()},
            {"params": model.traj_gat_hidden2pos.parameters()},
            {"params": model.pred_lstm_model.parameters()},
            {"params": model.pred_hidden2pos.parameters()},
        ],
        lr=args.lr,
    )

    global best_ade
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            # best_ade = checkpoint["best_ade"]
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    training_step = 1

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        if epoch < 300:
            training_step = 1
        elif epoch < 500:
            training_step = 3
        else:
            training_step = 4

        train(args, model, train_loader, optimizer, epoch, training_step)
        if training_step == 4:
            ade = validate(args, model, val_loader)
            is_best = ade < best_ade
            best_ade = min(ade, best_ade)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_ade": best_ade,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )


def train(args, model, train_loader, optimizer, epoch, training_step):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    for batch_idx, batch in enumerate(train_loader):
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
        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len :]

        if training_step == 1 or training_step == 3:
            model_input = obs_traj_rel
            pred_traj_fake_rel = model(
                model_input, obs_traj, seq_start_end, 1, False, training_step
            )
            l2_loss_rel.append(
                l2_loss(pred_traj_fake_rel, model_input, loss_mask, mode="raw")
            )
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-4
            model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            for _ in range(args.best_k):
                pred_traj_fake_rel = model(model_input, obs_traj, seq_start_end, 0)
                l2_loss_rel.append(
                    l2_loss(
                        pred_traj_fake_rel,
                        model_input[-args.pred_len :],
                        loss_mask,
                        mode="raw",
                    )
                )

        l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
            _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
            _l2_loss_rel = torch.min(_l2_loss_rel) / (
                (pred_traj_fake_rel.shape[0]) * (end - start)
            )
            l2_loss_sum_rel += _l2_loss_rel

        loss += l2_loss_sum_rel
        losses.update(loss.item(), obs_traj.shape[1])
        loss.backward()
        optimizer.step()

        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)


def validate(args, model, val_loader):
    ade = utils.AverageMeter("ADE", ":.6f")
    fde = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
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
            loss_mask = loss_mask[:, args.obs_len :]
            # model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim = 0)
            pred_traj_fake_rel = model(obs_traj_rel, obs_traj, seq_start_end)

            pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len :]
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[-1])
            ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
            ade_ = ade_ / (obs_traj.shape[1] * args.pred_len)

            fde_ = fde_ / (obs_traj.shape[1])
            ade.update(ade_, obs_traj.shape[1])
            fde.update(fde_, obs_traj.shape[1])

            if i % args.print_every == 0:
                progress.display(i)

        logging.info(
            " * ADE  {ade.avg:.3f} FDE  {fde.avg:.3f}".format(ade=ade, fde=fde)
        )
    return ade.avg


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        logging.info("-------------- lower ade ----------------")
        shutil.copyfile(filename, "model_best.pth.tar")


if __name__ == "__main__":
    args = parser.parse_args()
    utils.set_logger(os.path.join(args.log_dir, "train.log"))
    main(args)
