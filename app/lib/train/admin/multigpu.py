import torch.nn as nn
# Original upstream uses DistributedDataParallel (DDP) for multi-GPU. For Kaggle
# notebooks, DDP via torchrun is awkward, so we also accept nn.DataParallel here
# so save_checkpoint() correctly unwraps `.module` and avoids saving keys with
# the `module.` prefix that would later break inference-time loads.

def is_multi_gpu(net):
    return isinstance(net, (
        MultiGPU,
        nn.parallel.distributed.DistributedDataParallel,
        nn.DataParallel,
    ))


class MultiGPU(nn.parallel.distributed.DistributedDataParallel):
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)