import torch

# Evaluate gradient of input with respect to function f
def evaluate(func, x):
    x.requires_grad_()
    y = func(x)
    y.backward()
    temp = x.grad
    x.grad = None
    x.requires_grad_(False)
    return temp

def criteria(func, current, next, epsilon, rule):
    
    if rule == "gradNorm":
        grad_next = evaluate(func, next)
        return torch.norm(grad_next) >= epsilon
    elif rule == "valueDiff":
        return (func(current) - func(next)) >= epsilon
    elif rule == "stepLen":
        return torch.norm(current - next) >= epsilon
