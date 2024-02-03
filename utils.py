import numpy as np
import torch
from itertools import product
from sympy import Symbol, sympify
def target_loss(pred,answer):
	pred = pred[:,0]
	
	return torch.mean(torch.sum((pred - answer)**2), dim=0)
def data_gen(formulas, n_formula, size, t_max, t):
    formula_data = formulas["formulae"][n_formula]
    formula = sympify(formula_data["formula"])
    sym_vars = {i[0] : Symbol(i[0]) for i in formula_data["vars"]}
    vars = {}
    for var in formula_data["vars"]:
        if var[1] == "log":
            vars[sym_vars[var[0]]] = np.random.lognormal(var[-2], var[-1], size)
        elif var[1] == "normal":
            vars[sym_vars[var[0]]] = np.random.normal(var[-2], var[-1], size)
    combinations = product(*vars.values())
    combinations = [dict(zip(vars.keys(), values)) for values in combinations]
    results = []
    for combination in combinations:
        formula_no_time = float(formula.subs(combination).evalf())
        formula_output = list(formula_no_time * t)
        t_pred = np.random.uniform(0, t_max)
        pred = formula_no_time * t_pred
        result = {}
        result["vars"] = list(combination.values())
        result["timeserie"] = formula_output
        result["question"] = t_pred
        result["answer"] = pred
        results.append(result)
    return results