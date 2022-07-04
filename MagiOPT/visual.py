import matplotlib.pyplot as plt
import torch

def plot_surface(func, sequence, info=None):
    Max, shape = window(sequence)
    fig = plt.figure(figsize=(8, 8))
    
    if shape == 1:
        ax = fig.add_subplot(111)
        X = torch.linspace(-Max, Max, 250)
        Y = func(X)
        y = func(sequence)
        
        ax.plot(X.numpy(), Y.numpy(), '-', color='b')
        ax.plot(sequence.numpy(), y.numpy(), color='r', marker='*')
        ax.set_xlabel("x")
        ax.set_ylabel("y")            
    else:
        ax = fig.add_subplot(111, projection='3d')
        x = torch.linspace(-Max, Max, 250)
        y = torch.linspace(-Max, Max, 250)
        
        X, Y = torch.meshgrid(x, y, indexing='xy')
        Z = func([X, Y])

        iter_x, iter_y = sequence[:,0], sequence[:,1]
        iter_z = func([iter_x, iter_y])
        
        ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='jet', alpha=.4)
        ax.plot(iter_x.numpy(), iter_y.numpy(), iter_z.numpy(), color='r', marker='*')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    if info != None:
        values = list(info.values())[0]
        if list(info.keys())[0] == 'al':
            ax.set_title(f"$\lambda=${values[0].item()}, $\sigma=${values[1].item()}")
        elif list(info.keys())[0] == 'pe':
            ax.set_title(f"$\sigma=${values[0].item()}")
        elif list(info.keys())[0] == 'ba':
            ax.set_title(f"$\mu=${values[0].item()}")
        else:
            raise ValueError
        
def plot_contour(func, sequence, info=None):
    Max, _ = window(sequence)

    x = torch.linspace(-Max, Max, 250)
    y = torch.linspace(-Max, Max, 250)
    
    X, Y = torch.meshgrid(x, y, indexing='xy')
    Z = func([X, Y])

    iter_x, iter_y = sequence[:, 0], sequence[:, 1]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111)
    ax.contour(X.numpy(), Y.numpy(), Z.numpy(), 90, cmap='jet')
    ax.plot(iter_x.numpy(), iter_y.numpy(), color='r', marker='*')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    if info != None:
        values = list(info.values())[0]
        if list(info.keys())[0] == 'al':
            ax.set_title(f"$\lambda=${values[0].item()}, $\sigma=${values[1].item()}")
        elif list(info.keys())[0] == 'pe':
            ax.set_title(f"$\sigma=${values[0].item()}")
        elif list(info.keys())[0] == 'ba':
            ax.set_title(f"$\mu=${values[0].item()}")
        else:
            raise ValueError
    
    
def window(sequence):
    smax, _ = torch.max(sequence, axis=0)
    smin, _ = torch.min(sequence, axis=0)
    Max = 1.1 * max(max(smax), abs(min(smin)))
    shape = smax.shape[0]
    return Max, shape
