
<img src="https://github.com/MagiFeeney/MagiOPT/blob/1591476bc5f5d949b051a218b3608344accae686/logo/image/logo.png">

## Why MagiOPT?

- Unified framework for both uncontrained and constrained optimization
- Efficient and powered by PyTorch's automatic differentiation engine
- User-friendly and easily modifiable
- Visualization for both curve (or surface) and contour plots

## Installation
```
$ git clone ...
$ cd MagiOPT
```

## How to use
- Unconstrained
```python
import MagiOPT as optim
    
def func(x):
    ...
        
optimizer = optim.SD(func) # Steepest Descent
x = optimizer.step(x0)     # On-the-fly
optimizer.plot()           # Visualize
```

- Constrained
```python
import MagiOPT as optim
    
def object(x):
    ...
def constr1(x):
    ...
def constr2(x):
    ...
...
    
optimizer = optim.Penalty(object, 
                          sigma, 
                          (constr1, '<='), 
                          (constr2, '>='), 
                          plot=True)        # Penalty methoed
optimizer.BFGS()                            # Inner optimizer
x = optimizer.step(x0)                      # On-the-fly
```
## Supported Optimizers
| Unconstrained | Constrained |
| ------ | ------ |
| Steepest Descent | Penalty Method |
| Amortized Newton Method | Log-Barrier Method|
| SR1 | Inverse-Barrier Method |
| DFP | Augmented Lagrangian Method |
| BFGS | 
| Broyden |
| FR |
| PRP |
| CG for Qudratic Function |
| CG for Linear Equation |
| BB1 |
| BB2 |
| Gauss-Newton |
| LMF |
| Dogleg |

## Visualization
Use a simple line of code for unconstrained optimizer
```python
optimizer.plot()
```
we can visualize the 2D curve with iterated sequence, such that


| <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/sd/Figure_sd_1.png"> |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Or with the 3D surface with iterated sequence, and its contour with iterated sequence.
| <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/sd/Figure_sd_2.png">  |  <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/sd/Figure_sd_3.png"> |
|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|

Or use
```python
optimizer = optim.Penalty(..., plot=True)
```
we can visualize the function and sequence of each inner iteration with the 3D surface with iterated sequence, and its contour with iterated sequence.

| <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_1.png">  |  <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_2.png"> |
| ------ | ------ |

| <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_3.png">  |  <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_4.png"> |
| ------ | ------ |

| <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_5.png">  |  <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_6.png"> |
| ------ | ------ |

| <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_7.png">  |  <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_8.png"> |
| ------ | ------ |

| <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_9.png">  |  <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_10.png"> |
| ------ | ------ |

| <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_11.png">  |  <img src="https://github.com/MagiFeeney/MagiOPT/blob/859f80525d66a1d6799024721b8fb5a508a9ae6f/MagiOPT/examples/penalty/Figure_12.png"> |
| ------ | ------ |

## Reminder
- The majority of algorithms are sensitive to the initial point; choosing an appropriate starting point can save significant effort.
- In ill-conditioned situations, constrained optimizers may require trial and error.
- The Barzilai-Borwein method is unstable for non-quadratic problems; however, you can still infer the optimization path through intermediate visualizations.
- You can easily extract the optimization sequence using:
  ```python
  optimization.sequence
  ```
- Your function should be supported by PyTorch operations; however, the input doesn't have to be. It can be a NumPy array, PyTorch tensor, or even a list.
## Requirements
- Pytorch 3.7 or above
