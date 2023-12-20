import itertools
from tqdm import tqdm
from tools import *
def generate_minimal_loss_formula(formuler,features, features_, labels, device, criterion, model, prob_size=4):
    _, formula, probas, _ = formuler.forward_test(features_.to(device))
    formula_space = []
    
    for m in range(len(probas)):
        proba = probas[m]
        space = [form(proba, m, model, bin=(m % 2 != 0), k=k) for k in range(prob_size)]
        s = []    
        for i in range(len(space[0])):
            op_ = [s_[i] for s_ in space]
            op_ = list(set(tuple(inner_list) for inner_list in  op_))
            op_ = [list(inner_tuple) for inner_tuple in op_]
            s.append(op_)
        formula_space.append(s)
    
    losses = []
    for_ = []

    for vector in tqdm(itertools.product(range(prob_size), repeat=sum(model[1:])), desc="Processing Vectors", unit="vector"):
        _f = [
            [
                formula_space[i][j][split_vector(vector, model[1:])[i][j]]
                for j in range(len(split_vector(vector, model[1:])[i]))
            ]
            for i in range(len(formula_space))
        ]
        out = decode(_f, features.to(device))
        loss_f = criterion(out.squeeze(1), labels.to(device)).detach().cpu()
        losses.append(float(loss_f))
        for_.append(_f)
    
    min_loss = min(losses)
    min_for = for_[losses.index(min_loss)]
    
    return min_loss, min_for
