"""
描述：这里一个简单的分布式教程，使用它可以轻易的拓展您的算法已实现分布式训练加速
"""

import os
import sys
import math
import tempfile

import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import Dataset

# 初始化分布式训练模式
def init_distributed_mode(args):
    """
    rank：用于表示进程的编号/序号（在一些结构图中rank指的是软节点，rank可以看成一个计算单位），
    每一个进程对应了一个rank的进程，整个分布式由许多rank完成。
    word size：全局（一个分布式任务）中，rank的数量。
    gpu：分布式训练用的GPU个数。
    """

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


# 资源释放
def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# 获取分布式训练中rank的数量
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# 判断当前进程是否为主进程。一般在主进程中打印log
def is_main_process():
    return get_rank() == 0


# 计算所有rank上value变量的平均值，并返回
def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value



###############example#################################
"""
下面是一个分布式训练的例子，来介绍如何在你的目标检测算法中实施分布式训练
"""


class MyDataSet(Dataset):
    """
    简单的例子，自己实现时请重写__getitem__()和__len__()
    """
    def __init__(self, images_path: list, images_class: list, transform=None):
        super(MyDataSet, self).__init__()
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class MyModel(nn.Module):
    """
    简单示例，请重写forward()方法
    """
    def __init__(self):
        super(MyModel, self).__init__()
        pass

    def forward(self):
        pass


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item()


def main(args):
    # step1:初始化各进程环境
    init_distributed_mode(args=args)
    #赋值，其实也可以直接调用args.rank
    rank = args.rank
    device = torch.device(args.device)
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)

    #step2:实例化data_loader，注意采用分布式训练dataset跟单卡训练时的略有不同
    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=None,
                               images_class=None,
                               transform=None)

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=None,
                             images_class=None,
                             transform=None)

    # 给每个rank对应的进程分配训练的样本索引,这里非常重要
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    # 注意rank==0代表主进程
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=args.batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)

    # step 3：实例化模型，并构建分布式DDP模型
    model = MyModel().to(device)

    # 下面部分用于载入预训练权重，以及一些其他的操作
    # 如果存在预训练权重则载入
    if os.path.exists(args.weights_path):
        weights_dict = torch.load(args.weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # step4 ：开始训练，这里只展示训练时的为伪代码，关于一些优化器初始化，loss函数实例化则跳过
    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
        acc = sum_num / val_sampler.total_size

        # 请注意这里，在主进程中打印输出，并保存结果
        if rank == 0:
            print("[epoch {}] accuracy: {}， mean loss: {}".format(epoch, round(acc, 3), round(mean_loss, 3)))
            if args.save_best:
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.module.state_dict(), "./weights/best-model-{}.pth".format(epoch))
            else:
                torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))
        # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    cleanup()

# 启动接口
if __name__ == '__main__':

    main()


########分布式启动命令########
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env distributed_demo.py #这后面可以跟你设置的需要载入超参数

