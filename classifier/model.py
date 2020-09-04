import torch
import torch.nn as nn
from torchvision import models

class Model:
    def build_model(self):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 2)

        return model

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
        return model

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
