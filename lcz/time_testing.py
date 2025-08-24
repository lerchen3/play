import time
import csv
from itertools import product
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)




class LCZNet(nn.Module):
    def __init__(self, in_channels, num_res_blocks, num_filters, action_size=20480):
        super().__init__()
        self.conv_init = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(num_filters)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        self.conv_policy = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, action_size)

        self.conv_value = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1 * 8 * 8, num_filters)
        self.fc_value2 = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks:
            x = block(x)
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(x.size(0), -1)
        p = F.log_softmax(self.fc_policy(p), dim=1)
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(x.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))
        return p, v


def measure_time(in_ch, n_blocks, n_filters, batch_size, device):
    net = LCZNet(in_ch, n_blocks, n_filters).to(device)
    net.train()
    x = torch.randn(batch_size, in_ch, 8, 8, device=device)
    torch.cuda.synchronize(device) if device != "cpu" else None
    start_f = time.time()
    p, v = net(x)
    loss = p.sum() + v.sum()
    torch.cuda.synchronize(device) if device != "cpu" else None
    end_f = time.time()
    start_b = time.time()
    loss.backward()
    torch.cuda.synchronize(device) if device != "cpu" else None
    end_b = time.time()
    print(f"Params: in_ch={in_ch}, n_blocks={n_blocks}, n_filters={n_filters}, batch_size={batch_size}")
    print(f"Forward time: {end_f - start_f:.4f}s, Backward time: {end_b - start_b:.4f}s")
    return end_f - start_f, end_b - start_b


def main():
    parser = argparse.ArgumentParser(description="Measure network throughput")
    parser.add_argument("--in_channels", type=int, default=103)
    parser.add_argument("--num_res_blocks", type=int, default=40)
    parser.add_argument("--num_filters", type=int, default=256)
    parser.add_argument(
        "--batch_sizes", type=int, nargs="+",
        default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels_list = [args.in_channels]
    num_res_blocks_list = [args.num_res_blocks]
    num_filters_list = [args.num_filters]
    batch_sizes = args.batch_sizes

    rows = []
    for in_ch, n_blocks, n_filters, bsz in product(in_channels_list, num_res_blocks_list, num_filters_list, batch_sizes):
        fwd, bwd = measure_time(in_ch, n_blocks, n_filters, bsz, device)
        rows.append([in_ch, n_blocks, n_filters, bsz, fwd, bwd])

    with open("time_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["in_channels", "num_res_blocks", "num_filters", "batch_size", "forward_time", "backward_time"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
