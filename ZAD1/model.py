from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.hidden = nn.ModuleList()
        self.output = self.generate_layers()

    def generate_layers(self):
        layer_switch = {
            5: [2, 4, 5, 8, 10],
            4: [3, 6, 9, 12],
            3: [5, 10, 15],
            2: [10, 20],
            1: [30]
        }
        layers_n = layer_switch.get(self.layer, [200])

        input_n = 9417
        for x in layers_n:
            self.hidden.append(nn.Linear(input_n, x * 32))
            input_n = x * 32
        return nn.Linear(input_n, 32)

    def forward(self, x):
        for hidden in self.hidden:
            x = F.relu(hidden(x))
        x = F.log_softmax(self.output(x), dim=1)
        return x
