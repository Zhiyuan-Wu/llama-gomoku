import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from llama import Llama
import json
import argparse
import os
import tqdm
import loralib
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/pretrain",
        help="The directory where the data file is stored.",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        default="model",
        help="The path to save model.",
    )
    parser.add_argument(
        "--saved_name",
        type=str,
        default="projection",
        help="The file name to save model.",
    )
    parser.add_argument(
        "--epoch_num",
        type=int,
        default=2,
        help="th",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of requests to send to Server at a time."
    )
    return parser.parse_args()

class multi_modal_dataset(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        data_files = sorted(Path(dataset_dir).glob("*.json"))
        self.instances = []
        for data_file in data_files:
            with open(data_file) as fin:
                lines = fin.readlines()
                for line in lines:
                    data = json.loads(line)
                    self.instances.append((data["canonicalBoard"], data["prompt"], data["completion"]))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

if __name__ == "__main__":
    args = parse_args()
    dataset = multi_modal_dataset(args.dataset_dir)
    dataset = DataLoader(dataset, batch_size=args.batch_size)
    model = Llama.build(
                        ckpt_dir="/home/zhiyuan/llama2/llama/llama-2-7b",
                        lora_path=None,
                        tokenizer_path="/home/zhiyuan/llama2/llama/tokenizer.model",
                        GobangNNet_path="/home/zhiyuan/talking_alpha_zero/model/checkpoint_n15b6c128i3000d230912.pth",
                        max_seq_len=2048,
                        max_batch_size=args.batch_size,
                        )
    model.set_require_grad(text=False, text_lora=False, projection=True, policy=False)
    total_para = 0
    trainable_para = 0
    for name, para in model.model.named_parameters():
        total_para += para.nelement()
        if para.requires_grad:
            trainable_para += para.nelement()
    print(f"total: {total_para}, tainable: {trainable_para}, ratio: {trainable_para/total_para}")
            
    optimizer = torch.optim.SGD(model.model.parameters(), lr=1e-4, )
    # optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4, eps=1e-4)

    progress_bar = tqdm.tqdm(total=args.epoch_num * len(dataset))
    loss_ma = 0
    model.model.train()
    for e in range(args.epoch_num):
        for batch, (b, p, t) in enumerate(dataset):
            loss = model.compute_loss(b, p, t)
            loss_ma = 0.999*loss_ma + 0.001*loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss_ma)
            progress_bar.update(1)
            
        torch.save(loralib.lora_state_dict(model.model), os.path.join(args.saved_path, f"{args.saved_name}_e{e}.pth"))
            