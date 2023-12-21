import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
operators_uni = [
    "sin", "cos", "sqrt", "", "square", "zeroise", "pi", "g", "ln", "abs"
]
operators_bin = ["+", "-", "*", "/"]






def standardize_tensor(input_tensor, reference_tensor):
    min_val = torch.min(reference_tensor)
    max_val = torch.max(reference_tensor)
    
    normalized_tensor = (input_tensor - min_val) / (max_val - min_val + 1e-6)
    
    return input_tensor

def decode(formula, x):
    for i in range(len(formula)):
        out = torch.Tensor([]).to(device)
        for f in formula[i]:
            if (i % 2 == 0):
                out_ =  uni_(f, x).unsqueeze(1)
            else:
                out_ =  bin_(f, x).unsqueeze(1)
            out = torch.cat((out, out_), 1)
        x = out
    return x
def filter_k_largest(tensor, k):
    # Get the K largest values along the third axis
    k_largest_values, _ = torch.topk(tensor, k, dim=2, largest=True, sorted=True)

    # Create a mask where values less than the Kth largest are set to 0
    mask = tensor < k_largest_values[:, :, -1].unsqueeze(2)

    # Apply the mask to zero out values less than the Kth largest
    filtered_tensor = tensor.masked_fill(mask, 0)

    return filtered_tensor
def replicate_mean(tensor, k):
    # Calculate the mean along the first axis
    mean = torch.mean(tensor, dim=0, keepdim=True)
    # Apply the mask to zero out values less than the Kth largest
    filtered_mean = filter_k_largest(mean, k)
    # Replicate the mean to create a tensor of the same size as the original tensor
    replicated_mean = filtered_mean.repeat(tensor.shape[0], 1, 1)
    return replicated_mean

def data_prep(dataset, part, n_equation, equations_size):
    sub_data = dataset[part]["text"]
    # n represents the number of samples per equation
    n = int(len(sub_data) / equations_size)
    equation_data = sub_data[n * (n_equation - 1): n * n_equation]
    equation_data = [[float(j) for j in i.split(" ")] for i in equation_data]
    equation_data = np.array(equation_data)
    x, y = equation_data[:, :-1], equation_data[:, -1]
    return x, y

def combine(x, bin=True):
    if bin:
        return len(operators_bin) * x ** 2 - (len(operators_bin) - 1) * x
    else:
        return len(operators_uni) * x 
def var_gen(n_var):
    return [f"x{i}" for i in range(n_var)]
def uni_gen(n_var):
    var = [i for i in range(n_var)]
    op = []
    for _i in var:
        for _j in operators_uni:
                op.append([_j, _i])
    return op
    
def bin_gen(n_var):
    variables = [i for i in range(n_var)]
    op = []
    for i_ in operators_bin:
        for j_ in variables:
            for k_ in variables:
                if (i_ == "/") and (j_ == k_):
                    pass
                elif (i_ == "-") and (j_ == k_):
                    pass
                elif (i_ == "+") and (j_ == k_):
                    pass
                elif (i_ == "*") and (j_ == k_):
                    pass
                else:
                    op.append([j_, i_, k_])
    op += [[i] for i in variables]
    return op

def bin_(op, x):
    if len(op) == 3:
        a, op, b = op[0], op[1], op[2]
        if op == "/":
            return x[:, a] / (x[:, b] + (x[:, b] / torch.abs(x[:, b]) * 1e-6))
        elif op == "+":
            return x[:, a] + x[:, b]
        elif op == "*":
            return x[:, a] * x[:, b]
        elif op == "-":
            return x[:, a] - x[:, b]
    else:
        return x[:, op[0]]
def uni_(op, x):
    op, a = op[0], op[1]
    if op == "cos":
        return torch.cos(x[:, a])
    elif op == "sin":
        return torch.sin(x[:, a])
    elif op == "abs":
        return torch.abs(x[:, a])
    elif op == "ln":
        return torch.log(torch.abs(x[:, a])+ 1e-3)
    elif op == "sqrt":
        return torch.sqrt(torch.abs(x[:, a]))
    elif op == "exp":
        return torch.exp(x[:, a])
    elif op == "square":
        return torch.square(x[:, a])
    elif op == "":
        return x[:, a]
    elif op == "pi":
        return torch.pi * x[:, a]
    elif op == "g":
        return 9.807 * x[:, a]
    elif op=="zeroise":
        return torch.zeros_like(x[:,a]) + 1e-5

def select(x, var, current_var,  bin=True):
    if bin:
        operations = bin_gen(current_var)
    else:
        operations = uni_gen(current_var)
    output_var = torch.Tensor([]).to(device)
    for op in operations:
        if bin:
            out = bin_(op, x)
        else:
            out = uni_(op, x)
        if torch.isinf(out).any().item() or torch.isnan(out).any().item():
            print(x)
            print(op)
            break
        output_var = torch.cat((output_var, out.unsqueeze(1)), 1)
    output_var = output_var.unsqueeze(1).expand(-1, var, -1)
    return output_var, operations
def split_vector(vector, shape):
    result = []
    start_index = 0
    for size in shape:
        end_index = start_index + size
        result.append(vector[start_index:end_index])
        start_index = end_index
    return result
def real_grad(x):
    with torch.enable_grad():
        x = x.clone().detach().requires_grad_(True)
        # Perform the operation
        A = x[:, 0] * x[:, 1]

    # Create a new tensor for backpropagation with gradients
    gradient_tensor = torch.ones_like(A, requires_grad=True)

    # Perform backward pass with respect to A using the new tensor
    A.backward(gradient_tensor, retain_graph=True)

    # The gradient of A with respect to C
    grad_ = x.grad
    return grad_
def compute_gradient_last_to_first(model, input_data):
    input_data.requires_grad = True  # Set requires_grad to True to compute gradients
    output, _ = model(input_data)
    gradient = torch.autograd.grad(outputs=output, inputs=input_data, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
    return gradient
def importance(B, C):
    with torch.enable_grad():
        C = C.clone().detach().requires_grad_(True)
        # Perform the operation
        A = torch.sum(B * C, axis=2)

    # Create a new tensor for backpropagation with gradients
    gradient_tensor = torch.ones_like(A, requires_grad=True)

    # Perform backward pass with respect to A using the new tensor
    A.backward(gradient_tensor, retain_graph=True)

    # The gradient of A with respect to C
    grad_ = C.grad
    return grad_
def form(proba, i, model, bin=True, k=0):
    pr_ = torch.mean(proba, dim=0)
    if bin:
        vars = bin_gen(model[i])
    else:
        vars = uni_gen(model[i])
    idx = []
    for c in range(model[i + 1]):
            idx.append(torch.argsort(pr_[c], descending=True)[k])
    formula = [vars[l.detach().cpu()] for l in idx]
    return formula