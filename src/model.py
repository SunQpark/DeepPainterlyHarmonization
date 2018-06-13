import torch
import torch.nn as nn
from torchvision.models import vgg16_bn

class FeatureExtracter(nn.Module):
    def __init__(self):
        super(FeatureExtracter, self).__init__()
        model = vgg16_bn(pretrained=True).features
        indices = [0, 3, 10, 17, 27, 37]
        self.layers = self._slice_layers(model, indices) 
    
    def forward(self, x):
        out = []
        for l in self.layers:
            x = l(x)
            out.append(x)
        return out
        
    def _slice_layers(self, model, indices):
        layers = []
        for i in range(len(indices) - 1):
            layer = nn.Sequential(*list(model.children())[indices[i]:indices[i + 1]])
            layers.append(layer)
        return nn.Sequential(*layers)
        

if __name__ == '__main__':
    model = FeatureExtracter()
    with torch.no_grad():
        dummy_input = torch.rand(2, 3, 224, 224)
        print(list(model.children()))
        for i, output in enumerate(model(dummy_input)):
            print("shape of target_layer {} : ".format(i+1), output.shape)
