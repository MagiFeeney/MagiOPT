def backforth(func, x, d, a, b, step):

    def phi(alpha):
        return func(x + alpha * d)
    
    a0 = (b + a) / 2
    gamma = 1.2 
    
    a1 = a0 + step
    count = 0
    while (phi(a1) < phi(a0) or count == 0):
        a = a0
        if phi(a1) >= phi(a0) and count == 0:
            step = -step
        else:
            step = step * gamma
        a0 = a1
        a1 = a0 + step
        count += 1
        if a1 < 0:
            a1 = 0
            break

    left = min(a, a1)
    right = max(a, a1)

    return left, right
   


