import os
import re
import sys
import torch
import numpy as np
import editdistance
from collections import OrderedDict

from data import dataset
from model import HTR_VT
from utils import utils
import valid


def load_model(pth_path, nb_cls, img_size):
    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=img_size[::-1])
    ckpt = torch.load(pth_path, map_location='cpu')
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in ckpt['state_dict_ema'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v
    model.load_state_dict(model_dict, strict=True)
    model = model.cuda()
    model.eval()
    return model


def evaluate_split(model, split_name, list_path, data_path, img_size, ralph,
                   converter, criterion, val_bs, save_dir):
    print(f"\n===== {split_name} =====")
    ds = dataset.myLoadDS(list_path, data_path, img_size, ralph=ralph)
    loader = torch.utils.data.DataLoader(ds, batch_size=val_bs, shuffle=False,
                                         pin_memory=True, num_workers=0)

    val_loss, val_cer, val_wer, preds, labels = valid.validation(
        model, criterion, loader, converter)

    print(f"CER  : {val_cer:.4f}  ({val_cer * 100:.2f}%)")
    print(f"WER  : {val_wer:.4f}  ({val_wer * 100:.2f}%)")
    print(f"Loss : {val_loss:.4f}")
    print(f"N    : {len(ds)}")

    edits, lengths = [], []
    for pred, gt in zip(preds, labels):
        edits.append(editdistance.eval(pred, gt))
        lengths.append(len(gt))
    edits = np.array(edits, dtype=np.int32)
    lengths = np.array(lengths, dtype=np.int32)

    npz_path = os.path.join(save_dir, f"bootstrap_{split_name}.npz")
    np.savez(npz_path, edits=edits, lengths=lengths)
    print(f"Saved: {npz_path}")
    print(f"Total edits / chars: {edits.sum()} / {lengths.sum()} "
          f"(recomputed CER: {edits.sum() / lengths.sum():.4f})")

    return {"cer": val_cer, "wer": val_wer, "loss": val_loss, "n": len(ds)}


def main():
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "mphd_htrvt_base_early_seed123"
    out_dir    = "./output"
    data_path  = "./data/persian/lines/"
    train_list = "./data/persian/train.ln"
    save_dir   = os.path.join(out_dir, exp_name)

    nb_cls   = 59
    img_size = [512, 64]
    val_bs   = 8

    pth_path = os.path.join(save_dir, "best_CER.pth")
    print(f"Loading checkpoint from: {pth_path}")
    if not os.path.exists(pth_path):
        print("Checkpoint not found!")
        return

    model = load_model(pth_path, nb_cls, img_size)
    print("Model loaded successfully.")

    train_dataset = dataset.myLoadDS(train_list, data_path, img_size)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)

    print(f"Charset size: {len(train_dataset.ralph)} (+ blank = {nb_cls})")

    splits = [
        ("test",    "./data/persian/test.ln"),
        ("test-t1", "./data/persian/test-t1.ln"),
        ("test-t2", "./data/persian/test-t2.ln"),
    ]

    results = {}
    for split_name, list_path in splits:
        results[split_name] = evaluate_split(
            model, split_name, list_path, data_path, img_size,
            train_dataset.ralph, converter, criterion, val_bs, save_dir)

    print("\n===== Summary =====")
    for split_name, r in results.items():
        print(f"{split_name:>8}: CER={r['cer']*100:.2f}%  WER={r['wer']*100:.2f}%  N={r['n']}")


if __name__ == '__main__':
    main()