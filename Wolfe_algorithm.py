import numpy as np
import sympy


def find_max_index(x, n: int=2):
    '''
    x: numpy array
    n: int
    index: numpy array
    '''
    x_ = x.copy()
    all_idx = np.arange(0, len(x)).tolist()
    index = []
    for _ in range(n):
        idx = np.argmax(x_)
        index.append(idx)
        x_[idx] = np.min(x)
        all_idx.remove(idx)
    return np.array(index).astype(np.int), np.array(all_idx).astype(np.int)


def wolfe(fx, x1, x2, A, x_init=[0, 0, 1, 1], eps=1e-6):
    dim = 4
    n = 2

    fg = 0 * x1
    grad = [sympy.diff(fx, x1), sympy.diff(fx, x2), fg, fg]
    # A = np.array([[1, 1, 1, 0], [1, 5, 0, 1]])
    x = x_init
    t = sympy.Symbol('t')
    p = np.ones(dim)

    
    while True:
        sub = {x1: x[0], x2: x[1]}
        max_index, non_index = find_max_index(x)
        B = A[:, max_index]
        N = A[:, non_index]
        grad_ = np.array([gradient.evalf(subs=sub) for gradient in grad])
        r = -1 * np.dot(np.dot(np.linalg.inv(B), N).transpose(), grad_[max_index]) + grad_[non_index]
        print('r\t', r)
        pN = [ -r[i] * x[idx] if r[i] > 0 else -r[i] for i, idx in enumerate(non_index)]
        pB = -1 * np.dot(np.dot(np.linalg.inv(B), N), pN)
        p[max_index] = pB
        p[non_index] = pN
        print('p\t', p)
        
        if np.linalg.norm(p) < eps:
            print('Final x is', x)
            break
        
        x_dir = [x[i] + t * p[i] for i in range(dim)]
        f_t = fx.subs({x1: x_dir[0], x2: x_dir[1]})
        t_max = np.min([- x[i] / p[i] if p[i] < 0 else np.Inf for i in range(dim)])
        t_val = one_dim(f_t, t, t_max)

        x = x + p * t_val
        print('current x is', x)
        print('y', fx.evalf(subs=sub))
        print('-' * 20)



def one_dim(f, t, t_max):
    df = sympy.diff(f)
    t_val = sympy.solve(df, t)
    t_val = t_val[0]
    if t_val > t_max:
        t_val = t_max
    print('best t is', t_val)
    return t_val



if __name__ == "__main__":
    
    x1 = sympy.Symbol('x1')
    x2 = sympy.Symbol('x2')
    fx = 2*x1**2 + 2*x2**2 - 2*x1*x2 - 4*x1 - 6*x2
    x_init = [0, 0, 2, 5]
    A = np.array([[1, 1, 1, 0], [1, 5, 0, 1]])
    print(find_max_index(x_init, 2))
    wolfe(fx, x1, x2, A, x_init)




    

