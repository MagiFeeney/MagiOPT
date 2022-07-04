ratio = 0.618
    
def gss(func, x, d, a, b, epsilon):

    def phi(alpha):
        return func(x + alpha * d)
    
    t1 = a + (1 - ratio) * (b - a)
    t2 = a + ratio * (b - a)

    while b - a > epsilon:
        if phi(t1) < phi(t2):
            b = t2
            t2 = t1
            t1 = a + (1 - ratio) * (b - a)
        else:
            a = t1
            t1 = t2
            t2 = a + ratio * (b - a)
        
    return (b + a) / 2
