from torch.nn.modules.loss import _Loss
import torch
# class PairWisePSA(_Loss):
#     def forward(self, est_targets, targets):
#         targets = targets.unsqueeze(1)
#         est_targets = est_targets.unsqueeze(2)
#         # print(targets.shape, est_targets.shape)
#         # (batch, 2, 2, time)
#         psa_loss_2 = targets * est_targets / torch.sqrt(torch.sum(targets * targets, -1)).unsqueeze(-1)
#         psa_loss = targets - psa_loss_2
#         psa_loss = torch.sqrt(torch.sum(psa_loss * psa_loss, axis=-1))
#         return psa_loss       

# class MultiSrcPSA
class SinglePSA(_Loss):
    def forward(self, est_targets, targets):
        psa_loss_2 = targets * est_targets / torch.sqrt(torch.sum(targets * targets, -1)).unsqueeze(-1)
        psa_loss = targets - psa_loss_2
        psa_loss = torch.sqrt(torch.sum(psa_loss * psa_loss, axis=-1))
        return psa_loss       

class PairWisePSA(_Loss):
    def forward(self, est_targets, targets, inputs):
        # est_targets -> a_hat
        # targets -> a
        # inputs -> y
        inputs = inputs.unsqueeze(1)
        targets = targets.unsqueeze(1)
        targets = torch.concat([targets, targets], axis=2)
        est_targets = est_targets.unsqueeze(2)
        residual_cita = torch.nn.CosineSimilarity(dim=-1)(targets, inputs)
        cosine_cita = torch.cos(residual_cita)
        ret = est_targets - targets * cosine_cita
        psa_loss = torch.sqrt(torch.sum(ret * ret, axis=-1))
        return psa_loss
        # (batch, 2, 2, time)
