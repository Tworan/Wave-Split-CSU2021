import numpy as np
import torch
import sys
sys.path.append('/home/oneran/sudo_rm_rf')
from sudo_rm_rf.dnn.models.improved_sudormrf import SuDORMRF, UConvBlock

# load pretrained model
pretrained_model = torch.load('/home/oneran/sudo_rm_rf/Improved_Sudormrf_U16_Bases2048_WHAMRexclmark.pt')
backbone = list(pretrained_model.sm.children())[:8]
for i in range(8):
    for p in backbone[i].parameters():
        p.require_grads = False

for i in range(2):
    backbone.append(
        UConvBlock(
            out_channels=256,
            in_channels=512, 
            upsampling_depth=4
        )
    )
backbone_net = torch.nn.Sequential(*backbone)
# our model
def get_fusion_model():
    our_model = SuDORMRF(
        in_channels=512,
        out_channels=256, 
        num_blocks=12, 
        upsampling_depth=4,
        enc_kernel_size=41,
        enc_num_basis=512
    )
    our_model.sm = backbone_net
    return our_model