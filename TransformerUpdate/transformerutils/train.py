import argparse
import logging
import os
import random
import shutil
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import TransformerUpdate.utils
from TransformerUpdate.data.loader import data_loader
from models import Transformer
from TransformerUpdate.utils import (
    displacement_error,
    final_displacement_error,
    get_dset_path,
    int_tuple,
    l2_loss,
    relative_to_abs,
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
parser.add_argument("--batch_size", default=1, type=int)   # 原为64
parser.add_argument("--num_epochs", default=400, type=int)

parser.add_argument("--noise_dim", default=(8,), type=int_tuple)  # 原为16
parser.add_argument("--noise_type", default="gaussian")

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

parser.add_argument(
    "--lr",
    default=1e-5,  # 1*10(-3次方)
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)

parser.add_argument("--best_k", default=20, type=int)
parser.add_argument("--print_every", default=10, type=int)
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)

parser.add_argument(
    "--resume",
    # default="./checkpoint/checkpoint155.pth.tar",
    default="",
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
parser.add_argument('-n_layers', type=int, default=4)  # 因为加了一层initencoder改成2了
parser.add_argument('-dropout', type=float, default=0.1)

parser.add_argument('-embs_share_weight', action='store_true')
parser.add_argument('-proj_share_weight', action='store_true')

parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
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

    writer = SummaryWriter()

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
        n_gatheads=n_gatheads,n_gatunits=n_gatunits,alpha=args.alpha
    )

    model.cuda()
    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters()},
            {"params": model.decoder.parameters()},
            {"params": model.decoder_output_linear.parameters()}
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
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        train(args, model, train_loader, optimizer, epoch, writer)

        ade = validate(args, model, val_loader, epoch, writer)
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
            f"./checkpoint/checkpoint{epoch}.pth.tar",
        )
    writer.close()



def train(args, model, train_loader, optimizer, epoch, writer):
    losses = TransformerUpdate.utils.AverageMeter("Loss", ":.6f")
    progress = TransformerUpdate.utils.ProgressMeter(
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

        model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)

        for _ in range(args.best_k):
            pred_traj_fake_rel = model(obs_traj_rel,seq_start_end, pred_traj_gt_rel, istrain=0)  # 输入gt预测值

            l2_loss_rel.append(
                l2_loss(
                    pred_traj_fake_rel,
                    model_input[-args.pred_len:],  # [8,n,2]
                    loss_mask,
                    mode="raw",
                )
            )

        l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
            # torch.narrow(需切片数量，切片维度，开始的索引，切片长度)
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
    writer.add_scalar("train_loss", losses.avg, epoch)


def validate(args, model, val_loader, epoch, writer):
    ade = TransformerUpdate.utils.AverageMeter("ADE", ":.6f")
    fde = TransformerUpdate.utils.AverageMeter("FDE", ":.6f")
    progress = TransformerUpdate.utils.ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")

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
            pred_traj_fake_rel = model(obs_traj_rel,seq_start_end, pred_traj_gt_rel, istrain=0)

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
        writer.add_scalar("val_ade", ade.avg, epoch)
    return ade.avg


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")
        shutil.copyfile(filename, "model_best.pth.tar")


if __name__ == "__main__":
    args = parser.parse_args()
    TransformerUpdate.utils.set_logger(os.path.join(args.log_dir, "train.log"))
    checkpoint_dir = "./checkpoint"
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    main(args)
