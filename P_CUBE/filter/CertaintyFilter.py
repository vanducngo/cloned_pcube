import torch

class CertaintyFilter:
    def __init__(self, entropy_threshold):
        self.entropy_threshold = entropy_threshold
    def check(self, sample, model):
        with torch.no_grad():
            probs = torch.softmax(model(sample), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy < self.threshold, entropy