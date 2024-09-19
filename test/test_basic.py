import torchvision.models as models

from examples import *
import models._modules as my_nn


def test_get_lr_scheduler():
    args = argparse.Namespace()
    args.epochs = 100
    args.lr_scheduler = 'MultiStepLR'
    args.step_size = 20
    args.gamma = 0.1
    args.milestones = [10, 15, 40]
    args.lr = 1
    args.warmup_epoch = -1

    model = models.alexnet()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9)
    scheduler = get_lr_scheduler(optimizer, args)
    for i in range(args.epochs):
        scheduler.step()
        print('{}: {}'.format(i, optimizer.param_groups[0]['lr']))


if __name__ == '__main__':
    test_get_lr_scheduler()
