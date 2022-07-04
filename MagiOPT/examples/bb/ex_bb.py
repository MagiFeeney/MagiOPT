import MagiOPT as optim

def func(x):
  return 3 * x[0]**2 + 3 * x[1]**2 - x[0] * x[1]

def main():
    
    optimizer1 = optim.BB1(func)
    x1 = optimizer1.step([2, 2])
    optimizer1.plot()

    optimizer2 = optim.BB2(func)
    x2 = optimizer2.step([2, 2])
    optimizer2.plot()
    
if __name__ == "__main__":
    main()
