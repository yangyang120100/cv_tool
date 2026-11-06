import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)

    def forward(self, input):
        output = self.conv1(input)
        return output

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = MyModel().cuda()
    model = DDP(model, device_ids=[local_rank])

    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=bs,
    #     sampler=DistributedSampler(dataset)
    # )
    #
    # for epoch in range(epochs):
    #     train_loader.sampler.set_epoch(epoch)
    #     train_one_epoch(model, train_loader)
    #
    # dist.destroy_process_group()

if __name__ == '__main__':
    main()