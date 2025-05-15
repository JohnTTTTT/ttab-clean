# test_only.py

import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import timm
from timm.models.layers import trunc_normal_

# no more util.misc import — we don't want timestamped logs
#from util.misc import init_single_GPU_mode
from fmae_iat_files.datasets import build_AffectNet_dataset
from fmae_iat_files.pos_embed import interpolate_pos_embed
import models_vit_fmae
from fmae_iat_files.engine_finetune import evaluate


def get_args():
    parser = argparse.ArgumentParser('Eval fine-tuned ViT on AffectNet', add_help=False)
    parser.add_argument('--model',       default='vit_large_patch16', type=str)
    parser.add_argument('--input_size',  default=224,                type=int)
    parser.add_argument('--batch_size',  default=16,                 type=int)
    parser.add_argument('--finetune',    required=True,              type=str,
                        help='path to your finetuned .pth checkpoint')
    parser.add_argument('--test_path',   required=True,              type=str,
                        help='AffectNet validation folder')
    parser.add_argument('--nb_classes',  default=7,                  type=int,
                        help='number of emotion classes')
    parser.add_argument('--device',      default='cuda',             type=str)
    parser.add_argument('--seed',        default=0,                  type=int)
    parser.add_argument('--run_eval',    action='store_true',
                        help='if set, run evaluate() and print accuracy after dumping stats')
    return parser.parse_args()


def main(args):
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # build val dataloader
    dataset_val = build_AffectNet_dataset(args.test_path, is_train=False, args=args)
    loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    # build model
    model = models_vit_fmae.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=0.0,
        global_pool=True,
        grad_reverse=0,
    )

    # load checkpoint
    print(f"Loading checkpoint from {args.finetune}")
    ckpt = torch.load(args.finetune, map_location='cpu')
    state = ckpt.get('model', ckpt)

    print(f"[test_fmae] checkpoint keys ({len(state)}):")
    for k,v in state.items():
        print(f"  {k:40} -> {tuple(v.shape)}")

    model_sd = model.state_dict()
    for k in ('head.weight','head.bias'):
        if k in state and state[k].shape != model_sd[k].shape:
            print(f"  → dropping {k} (shape mismatch)")
            del state[k]
    interpolate_pos_embed(model, state)
    msg = model.load_state_dict(state, strict=False)
    print(f"Loaded state_dict: missing={msg.missing_keys}, unexpected={msg.unexpected_keys}\n")

    print(f"[test_fmae] load_state_dict missing keys:   {msg.missing_keys}")
    print(f"[test_fmae] load_state_dict unexpected keys: {msg.unexpected_keys}")

    print(f"[test_fmae] model.state_dict() ({len(model.state_dict())}):")
    for k,v in model.state_dict().items():
        print(f"  {k:40} -> {tuple(v.shape)}")

    model.to(device)

    if args.run_eval:
        stats = evaluate(loader_val, model, device)
        print(f"\nTest top-1 Accuracy: {stats['acc1']:.2f}%  (top-5: {stats['acc5']:.2f}%)")


if __name__ == '__main__':
    args = get_args()
    Path(args.finetune).expanduser().resolve()  # error early if bad path
    main(args)