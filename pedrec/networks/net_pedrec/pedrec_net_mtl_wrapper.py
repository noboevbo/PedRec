import torch.nn as nn


class PedRecNetMTLWrapper(nn.Module):
    def __init__(self, model: nn.Module, loss_head: nn.Module):
        super(PedRecNetMTLWrapper, self).__init__()
        self.model = model
        self.loss_head = loss_head

    def forward(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss_head(outputs, targets)
        return outputs, loss
