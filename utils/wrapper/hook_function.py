import torch.nn as nn

import models._modules as my_nn

__all__ = ['debug_graph_hooks']


def debug_graph(self, input, output):
    print('{}: type:{} input:{} ==> output:{} (max: {})'.format(self.name, type_str(self), [i.size() for i in input],
                                                                output.size(), output.max()))


def debug_graph_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, my_nn.Concat, nn.ZeroPad2d, my_nn.UpSample)):
            module.name = name
            module.register_forward_hook(debug_graph)


def type_str(module):
    if isinstance(module, nn.Conv2d):
        return 'Conv2d'
    if isinstance(module, nn.MaxPool2d):
        return 'MaxPool2d'
    if isinstance(module, nn.Linear):
        return 'Linear'
    if isinstance(module, nn.ZeroPad2d):
        return 'ZeroPad2d'
    return 'Emtpy'
