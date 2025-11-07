import torch

class ConsistencyFilter:
    def __init__(self, source_model):
        self.source_model = source_model
        
    def check(self, sample, current_model):
        with torch.no_grad():
            pred_current = current_model(sample).argmax()
            pred_source = self.source_model(sample).argmax()
        return pred_current == pred_source