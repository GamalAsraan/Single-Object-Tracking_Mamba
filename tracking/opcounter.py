from torchvision.models import resnet50
from thop import profile
import torch
from torch import nn
from lib.models.trackingmamba.trackingmamba import build_trackingmamba
from lib.config.trackingmamba.config import cfg, update_config_from_file

model = build_trackingmamba(cfg).cuda()
var2 = torch.randn(1, 3, 256, 256).cuda()
var1 = torch.randn(1, 3, 128, 128).cuda()


macs, params = profile(model, inputs=(var1, var2, ))
                        #custom_ops={YourModule: count_your_model})

# print(macs)
# print(params)
print('FLOPs = ' + str(macs / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')