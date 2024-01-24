# model.py
import torch
import torch.nn as nn
import torch.optim as optim
from tools import *
# Define your model classes and related functionalities here


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class Formuler(nn.Module):
    def __init__(self, model):
        super(Formuler, self).__init__()
        self.model = model
        self.encoders = nn.ModuleList()

        for i in range(len(model)-1):
                    odd = i % 2 != 0
                    in_dim = model[i]
                    out_dim = combine(model[i], bin=odd) * model[i + 1]
                    encoder = nn.Sequential(
                        nn.BatchNorm1d(in_dim),
                        nn.Linear(in_dim, out_dim * 4),
                        nn.ReLU(),
                        nn.Linear(out_dim * 4, out_dim),
                        nn.Linear(out_dim, out_dim, bias=False),
                        nn.Softmax(dim=1)
                    )
                    self.encoders.append(encoder)


    def forward(self, x):
        formula = []
        for i, encoder in enumerate(self.encoders):
            proba = encoder(x)
            proba = proba.view(-1, self.model[i + 1], proba.size()[1] // self.model[i + 1])
            operation_results, _ = select(x, self.model[i + 1], self.model[i], bin=(i % 2 != 0))
            x = (proba * operation_results).sum(dim=2)
            #importance_ = importance(proba, operation_results)
            importance_ = proba
            f = form(importance_, i, self.model, bin=(i % 2 != 0))
            formula.append(f)
        return x, formula
    def forward_kmax(self, x, k=2, replicate=True):
        formula = []
        probas = []
        for i, encoder in enumerate(self.encoders):
            proba = encoder(x)
            proba = proba.view(-1, self.model[i + 1], proba.size()[1] // self.model[i + 1])
            operation_results, _ = select(x, self.model[i + 1], self.model[i], bin=(i % 2 != 0))
            importance_ = importance(proba, operation_results)
            importance_ = filter_k_largest(importance_, k)
            if replicate == True:
                proba = replicate_mean(importance_, k)
            x = (importance_ * operation_results).sum(dim=2)
            f = form(importance_, i, self.model, bin=(i % 2 != 0))
            probas.append(importance_)
            formula.append(f)
        return x, formula, probas
    def forward_test(self, x):
            formula = []
            probas = []
            ops = []
            for i, encoder in enumerate(self.encoders):
                proba = encoder(x)
                proba = proba.view(-1, self.model[i + 1], proba.size()[1] // self.model[i + 1])
                operation_results, operations = select(x, self.model[i + 1], self.model[i], bin=(i % 2 != 0))
                x = (proba * operation_results).sum(dim=2)
                importance_ = importance(proba, operation_results)
                f = form(importance_, i, self.model, bin=(i % 2 != 0))
                ops.append(operations)
                formula.append(f)
                probas.append(importance_)
            return x, formula, probas, ops


