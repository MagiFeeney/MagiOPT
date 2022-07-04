import MagiOPT as optim

# 1D function
def func1(x):
    return x**2

# 2D function
def func2(x):
    return 3 * x[0]**2 + 3 * x[1]**2 - x[0] * x[1]

def main():
    optimizer1 = optim.SD(func1)
    x1 = optimizer1.step(2)
    optimizer1.plot()

    optimizer2 = optim.SD(func2)
    x2 = optimizer2.step([2, 5])
    optimizer2.plot()

if __name__ == "__main__":
    main()

    
    
