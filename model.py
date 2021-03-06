import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, layers):
        super(VGG, self).__init__()
        self.net = nn.Sequential(*layers)
        self.outputs = list()
        # We want to compute the style from conv1_1, conv2_1, conv3_1, conv4_1 and conv5_1
        # We want to compute the content from conv_4_1
        self.net[0].register_forward_hook(self._hook)
        self.net[5].register_forward_hook(self._hook)
        self.net[10].register_forward_hook(self._hook)
        self.net[19].register_forward_hook(self._hook)
        self.net[28].register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.outputs.append(output)

    def forward(self, x):
        _ = self.net(x)
        out = self.outputs.copy()
        self.outputs = list()
        return out
