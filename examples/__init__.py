import argparse
import datetime
import hashlib
import os
import random
import shutil
import time
import warnings

import google.protobuf as pb
import google.protobuf.text_format
import models._modules as my_nn
import numpy as np
import plotly.graph_objects as go
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
from proto import efficient_pytorch_pb2 as eppb
from pytorchcv.model_provider import get_model as ptcv_get_model
from tensorboardX import SummaryWriter
from utils import wrapper
from utils.ptflops import get_model_complexity_info
from warmup_scheduler import GradualWarmupScheduler
from torch.profiler import profile, record_function, ProfilerActivity
str_q_mode_map = {eppb.Qmode.layer_wise: my_nn.Qmodes.layer_wise,
                  eppb.Qmode.kernel_wise: my_nn.Qmodes.kernel_wise}


def get_base_parser():
    """
        Default values should keep stable.
    """

    print('Please do not import ipdb when using distributed training')

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--hp', type=str,
                        help='File path to save hyperparameter configuration')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume-after', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--bn-fusion', action='store_true', default=False,
                        help='ConvQ + BN fusion')
    parser.add_argument('--resave', action='store_true', default=False,
                        help='resave the model')

    parser.add_argument('--gen-layer-info', action='store_true', default=False,
                        help='whether to generate layer information for latency evaluation on hardware')

    parser.add_argument('--print-histogram', action='store_true', default=False,
                        help='save histogram of weight in tensorboard')
    parser.add_argument('--freeze-bn', action='store_true',
                        default=False, help='Freeze BN')
    return parser


def main_s1_set_seed(hp):
    if hp.HasField('seed'):
        random.seed(hp.seed)
        torch.manual_seed(hp.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def main_s2_start_worker(main_worker, args, hp):
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    args.world_size = hp.multi_gpu.world_size
    if hp.HasField('multi_gpu') and hp.multi_gpu.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or (hp.HasField(
        'multi_gpu') and hp.multi_gpu.multiprocessing_distributed)

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: {}'.format(ngpus_per_node))
    if hp.HasField('multi_gpu') and hp.multi_gpu.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def get_hyperparam(args):
    assert os.path.exists(args.hp)
    hp = eppb.HyperParam()
    with open(args.hp, 'r') as rf:
        pb.text_format.Merge(rf.read(), hp)
    print(hp)
    
    return hp


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2])
                        for x in open('tmp', 'r').readlines()]
    os.system('rm tmp')
    # TODO; if no gpu, return None
    try:
        visible_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        memory_visible = []
        for i in visible_gpu.split(','):
            memory_visible.append(memory_available[int(i)])
        return np.argmax(memory_visible)
    except KeyError:
        return np.argmax(memory_available)


def get_lr_scheduler(optimizer, lr_domain):
    """
    Args:
        optimizer:
        lr_domain ([proto]): [lr configuration domain] e.g. args.hp args.hp.bit_pruner
    """
    if isinstance(lr_domain, argparse.Namespace):
        lr_domain = lr_domain.hp
    if lr_domain.lr_scheduler == eppb.LRScheduleType.CosineAnnealingLR:
        print('Use cosine scheduler')
        scheduler_next = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=lr_domain.epochs)
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.StepLR:
        print('Use step scheduler, step size: {}, gamma: {}'.format(
            lr_domain.step_lr.step_size, lr_domain.step_lr.gamma))
        scheduler_next = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_domain.step_lr.step_size, gamma=lr_domain.step_lr.gamma)
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.MultiStepLR:
        print('Use MultiStepLR scheduler, milestones: {}, gamma: {}'.format(
            lr_domain.multi_step_lr.milestones, lr_domain.multi_step_lr.gamma))
        scheduler_next = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_domain.multi_step_lr.milestones, gamma=lr_domain.multi_step_lr.gamma)
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.CyclicLR:
        print('Use CyclicLR scheduler')
        if not lr_domain.cyclic_lr.HasField('step_size_down'):
            step_size_down = None
        else:
            step_size_down = lr_domain.cyclic_lr.step_size_down

        cyclic_mode_map = {eppb.CyclicLRParam.Mode.triangular: 'triangular',
                           eppb.CyclicLRParam.Mode.triangular2: 'triangular2',
                           eppb.CyclicLRParam.Mode.exp_range: 'exp_range', }

        scheduler_next = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=lr_domain.cyclic_lr.base_lr, max_lr=lr_domain.cyclic_lr.max_lr,
            step_size_up=lr_domain.cyclic_lr.step_size_up, step_size_down=step_size_down,
            mode=cyclic_mode_map[lr_domain.cyclic_lr.mode], gamma=lr_domain.cyclic_lr.gamma)
    else:
        raise NotImplementedError
    if not lr_domain.HasField('warmup'):
        return scheduler_next
    print('Use warmup scheduler')
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=lr_domain.warmup.multiplier,
                                          total_epoch=lr_domain.warmup.epochs,
                                          after_scheduler=scheduler_next)
    return lr_scheduler


def get_optimizer(model, args):
    # define optimizer after process model
    print('define optimizer')
    if args.hp.optimizer == eppb.OptimizerType.SGD:
        params = add_weight_decay(model, weight_decay=args.hp.sgd.weight_decay,
                                  skip_keys=['expand_', 'running_scale', 'alpha',
                                             'standard_threshold', 'nbits'])
        optimizer = torch.optim.SGD(params, args.hp.lr,
                                    momentum=args.hp.sgd.momentum)
        print('Use SGD')
    elif args.hp.optimizer == eppb.OptimizerType.Adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.hp.lr, weight_decay=args.hp.adam.weight_decay)
        print('Use Adam')
    else:
        raise NotImplementedError
    return optimizer


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def add_weight_decay(model, weight_decay, skip_keys):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        #print(name)
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in skip_keys:
            if skip_key in name:
                no_decay.append(param)
                added = True
                break
        if not added:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def set_bn_eval(m):
    """[summary]
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
    https://github.com/pytorch/pytorch/issues/16149
        requires_grad does not change the train/eval mode, 
        but will avoid calculating the gradients for the affine parameters (weight and bias).
        bn.train() and bn.eval() will change the usage of the running stats (running_mean and running_var).
    For detailed computation of Batch Normalization, please refer to the source code here.
    https://github.com/pytorch/pytorch/blob/83c054de481d4f65a8a73a903edd6beaac18e8bc/torch/csrc/jit/passes/graph_fuser.cpp#L232
    The input is normalized by the calculated mean and variance first. 
    Then the transformation of w*x+b is applied on it by adding the operations to the computational graph.
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
    return


def set_bn_grad_false(m):
    """freeze \gamma and \beta in BatchNorm
        model.apply(set_bn_grad_false)
        optimizer = SGD(model.parameters())
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if m.affine:
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)


def set_param_grad_false(model):
    for name, param in model.named_parameters():  # same to set bn val? No
        if param.requires_grad:
            param.requires_grad_(False)
            print('frozen weights. shape:{}'.format(param.shape))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.hp.print_freq == 0:
                progress.print(i)
            if args.hp.overfit_test:
                break

        print(' *Time {time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(time=batch_time, top1=top1, top5=top5))

    return top1.avg, top5.avg


def update_scale_param(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    
    ###starting from scratch. Not resuming
    for name, param in model.named_parameters():
        if 'flag' in name:
            param.data.copy_(torch.zeros(1))
        if 'alpha_cim' in name:
            continue
        param.requires_grad = False
    
    # switch to train mode
    model.train()
    
    end = time.time()
    base_step = epoch * args.batch_num
    
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        # compute output
        
        
        output = model(inputs)
        
        
        loss = criterion(output, targets)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        
        
        optimizer.step()
        
        # warning 1. backward 2. step 3. zero_grad
        # measure elapsed time
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        
        if (i % 20 == 0 and writer is not None) :
            for name, param in model.named_parameters():
                if 'alpha_cim' in name:
                    writer.add_histogram(name+'initialize',param.data,epoch* len(train_loader) + i)
                
                    if param.grad is not None:
                       
                        writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
        optimizer.zero_grad()
    
    for name, param in model.named_parameters():
        if 'flag' in name:
            param.data.copy_(torch.ones(1))
            continue
        
        param.requires_grad = True
    

def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    print('gpu id: {}'.format(args.gpu))
    # switch to train mode
    model.train()
    
    end = time.time()
    base_step = epoch * args.batch_num
    
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        # compute output
        
        
        output = model(inputs)
        
        
        loss = criterion(output, targets)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
        
        
        
        
        # compute gradient and do SGD step
        loss.backward()
        #optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        
        #print(len(optimizer.param_groups[0]['params']))
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        # warning 1. backward 2. step 3. zero_grad
        # measure elapsed time
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
        
        if (i % args.hp.log_freq == 0 and writer is not None) :
            for name, param in model.named_parameters():
                #print(name)
                writer.add_histogram(name,param.data,epoch* len(train_loader) + i)
                #if 'weight' in name:
                #    print(param.grad)
                if param.grad is not None:
                    writer.add_histogram(name+'/grad',param.grad.data,epoch* len(train_loader) + i)
        optimizer.zero_grad()
    return


def get_summary_writer(args, ngpus_per_node, model):
    if not args.hp.multi_gpu.multiprocessing_distributed or (args.hp.multi_gpu.multiprocessing_distributed
                                                             and args.hp.multi_gpu.rank % ngpus_per_node == 0):
        args.log_name = 'logger/{}_{}_{}'.format(args.hp.arch,
                                                 args.hp.log_name, get_current_time())
        writer = SummaryWriter(args.log_name)
        with open('{}/{}.prototxt'.format(args.log_name, args.arch), 'w') as wf:
            wf.write(str(args.hp))
        with open('{}/{}.txt'.format(args.log_name, args.arch), 'w') as wf:
            wf.write(str(model))
        return writer
    return None


def get_model_info(model, args, input_size=(3, 224, 224)):
    print('Inference for complexity summary')
    if isinstance(input_size, torch.utils.data.DataLoader):
        input_size = input_size.dataset.__getitem__(0)[0].shape
        input_size = (input_size[0], input_size[1], input_size[2])
    with open('{}/{}_flops.txt'.format(args.log_name, args.arch), 'w') as f:
        flops, params = get_model_complexity_info(
            model, input_size, as_strings=True, print_per_layer_stat=True, ost=f)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # with open('{}/{}.txt'.format(args.log_name, args.arch), 'w') as wf:
    #     wf.write(str(model))
    # with open('{}/{}.prototxt'.format(args.log_name, args.arch), 'w') as wf:
    #     wf.write(str(args.hp))
    # summary(model, input_size)
    if args.hp.export_onnx:
        dummy_input = torch.randn(1, input_size[0], input_size[1], input_size[2], requires_grad=True).cuda(args.gpu)
        # torch_out = model(dummy_input)
        torch.onnx.export(model,  # model being run
                          dummy_input,  # model input (or a tuple for multiple inputs)
                          # where to save the model (can be a file or file-like object)
                          "{}/{}.onnx".format(args.log_name, args.arch),
                          export_params=True,  # store the trained parameter weights inside the model file
                          # opset_version=10,  # the ONNX version to export the model to
                          input_names=['input'],  # the model's input names
                          output_names=['output']  # the model's output names
                          )
    return flops, params


def save_checkpoint(state, is_best, prefix, filename='checkpoint.pth.tar'):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'best.pth.tar')
    return


def process_model(model, args, replace_map=None, replace_first_layer=False, **kwargs_module):
    if not hasattr(args, 'arch'):
        args.arch = args.hp.arch

    if args.hp.HasField('weight'):
        if os.path.isfile(args.hp.weight):
            print("=> loading weight '{}'".format(args.hp.weight))
            weight = torch.load(args.hp.weight, map_location='cpu')
            model.load_state_dict(weight)
        else:
            print("=> no weight found at '{}'".format(args.hp.weight))

    if replace_map is not None:
        tool = wrapper.ReplaceModuleTool(model, replace_map, replace_first_layer, **kwargs_module)
        
        tool.replace()
        args.replace = [tool.convs, tool.linears, tool.acts]
        print('after modules replacement')
        display_model(model)
        info = ''
        for k, v in replace_map.items():
            if isinstance(v, list):
                for vv in v:
                    info += vv.__name__
            else:
                info += v.__name__
        args.arch = '{}_{}'.format(args.arch, info)
        print('Please update optimizer after modules replacement')

    if args.hp.HasField('resume'):
        if os.path.isfile(args.hp.resume):
            print("=> loading checkpoint '{}'".format(args.hp.resume))
            checkpoint = torch.load(args.hp.resume, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            
        else:
            print("=> no checkpoint found at '{}'".format(args.hp.resume))

    return


class DataloaderFactory(object):
    # MNIST
    mnist = 0
    # CIFAR10
    cifar10 = 10
    # ImageNet2012
    imagenet2012 = 40

    def __init__(self, args):
        self.args = args
        self.mnist_transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.cifar10_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.cifar10_transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def product_train_val_loader(self, data_type):
        args = self.args
        noverfit = not args.hp.overfit_test
        train_loader = None
        val_loader = None
        # MNIST
        if data_type == self.mnist:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(args.hp.data, train=True, download=True,
                                           transform=self.mnist_transform),
                batch_size=args.hp.batch_size, shuffle=True and noverfit)
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(args.hp.data, train=False, transform=self.mnist_transform),
                batch_size=args.hp.batch_size, shuffle=False)
            return train_loader, val_loader
        # CIFAR10
        if data_type == self.cifar10:
            trainset = torchvision.datasets.CIFAR10(root=args.hp.data, train=True, download=True,
                                                    transform=self.cifar10_transform_train)
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            else:
                train_sampler = None
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.hp.batch_size,
                                                       shuffle=(train_sampler is None) and noverfit,
                                                       num_workers=args.hp.workers, sampler=train_sampler)
            testset = torchvision.datasets.CIFAR10(root=args.hp.data, train=False, download=True,
                                                   transform=self.cifar10_transform_val)
            val_loader = torch.utils.data.DataLoader(testset, batch_size=args.hp.batch_size, shuffle=False,
                                                     num_workers=args.hp.workers)
            return train_loader, val_loader, train_sampler
        # ImageNet
        elif data_type == self.imagenet2012:
            # Data loading code
            traindir = os.path.join(args.hp.data, 'train')
            valdir = os.path.join(args.hp.data, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            train_dataset = torchvision.datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.hp.batch_size, shuffle=(train_sampler is None) and noverfit,
                num_workers=args.hp.workers, pin_memory=True, sampler=train_sampler, drop_last = True)

            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.hp.batch_size, shuffle=False,
                num_workers=args.hp.workers, pin_memory=True, drop_last=True)
            return train_loader, val_loader, train_sampler
        else:
            assert NotImplementedError
        return train_loader, val_loader


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.name = name
        self.fmt = fmt

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def distributed_model(model, ngpus_per_node, args):
    if not torch.cuda.is_available() or args.gpu is None:
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(int(args.gpu))
            model.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.hp.batch_size = int(args.hp.batch_size / ngpus_per_node)
            
            args.hp.workers = int(
                (args.hp.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
            #model = torch.nn.DataParallel(model).cuda()
        else:
            assert NotImplementedError
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(int(args.gpu))
        model = model.cuda(args.gpu)
    else:
        assert NotImplementedError
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.hp.arch.startswith('alexnet') or args.hp.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model


def get_hash_code(message):
    hash = hashlib.sha1(message.encode("UTF-8")).hexdigest()
    return hash[:8]


def get_current_time():
    currentDT = datetime.datetime.now()
    return currentDT.strftime("%Y-%m-%d-%H:%M")


def display_model(model):
    str_list = str(model).split('\n')
    if len(str_list) < 30:
        print(model)
        return
    begin = 10
    end = 5
    middle = len(str_list) - begin - end
    abbr_middle = ['...', '{} lines'.format(middle), '...']
    abbr_str = '\n'.join(str_list[:10] + abbr_middle + str_list[-5:])
    print(abbr_str)


def def_module_name(model):
    for module_name, module in model.named_modules():
        module.__name__ = module_name
