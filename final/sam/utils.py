from torch.nn.modules.batchnorm import _BatchNorm

def disable_running_stats(model):
    def _disable(module):
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                module.eval()

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                module.train()

    model.apply(_enable)