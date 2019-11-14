import numpy as np 
import sympy
import random


def f_r_method(fx, x1, x2, init, eps=1e-6):
    t = sympy.Symbol('t')

    x1_i = init['x1']
    x2_i = init['x2']
    sub = {x1: x1_i, x2: x2_i}
    dx1 = sympy.diff(fx, x1)
    dx2 = sympy.diff(fx, x2)
    grad_fx = sympy.sqrt(dx1 ** 2 + dx2 ** 2)
    sub = {x1: x1_i, x2: x2_i}
    p =  [-1 * dx1.evalf(subs = sub), -1 * dx2.evalf(subs = sub)]
    

    
    while grad_fx.evalf(subs= sub) >= eps:
        
        fx_ = fx.subs({x1: x1_i + p[0] * t, x2: x2_i + p[0] * t})
        t_i = one_dim_search(fx_, t)
        sub_ = {x1: x1_i + t_i * p[0], x2: x2_i + t_i * p[1]}

        grad_fx_i = grad_fx.evalf(subs=sub)
        grad_fx_i_ = grad_fx.evalf(subs=sub_)

        lmd = grad_fx_i_ ** 2 / grad_fx_i ** 2
        p_ = [-1 * dx1.evalf(subs=sub_) + lmd * p[0],
            -1 * dx2.evalf(subs=sub_) + lmd * p[1]]

        p = p_
        sub = sub_
        print('\n' * 2)
        print(sub)
        print(fx.evalf(subs=sub))
        print('\n' * 2)

    print('-' * 20 + '\n' + 'Finial:\n' )
    print(sub)
    print(fx.evalf(subs=sub))
        


def one_dim_search(fx, t, m1=0.3, m2 = 0.6, alpha= 2, eps=1e-3):
    t0 = 0
    a = 0
    b = 100
    # goldstein
    # random.randrange
    dfx = fx.diff(t)
    f = lambda x : fx.evalf(subs={t: x})
    df = lambda x : dfx.evalf(subs={t: x})

    tk = 0.5 * (a + b)

    while True:
        print('t\t', tk, end='\t')
        print('f\t', f(tk), end='\t')
        if abs(b - a ) < eps:
            break
        if f(tk) >= f(0) + m1 * tk * df(0):
            b = tk
            a = a
            tk = 0.5 * (a + b)
            continue
        if f(tk) <= f(0) + m2 * tk * df(0):
            a = tk
            b = b
            tk = .5 * (a + b)
            continue
        print('\n')
        print('t\t', tk)
        print('f\t', f(tk))
        break
    
        
    return tk


if __name__ == "__main__":
    x1 = sympy.Symbol('x1')
    x2 = sympy.Symbol('x2')
    init_value = {'x1': 0, 'x2': 0}
    fx = (1 - x1)**2 + 2 * (x2 - x1**2) ** 2
    t = sympy.Symbol('t')
    ft = ( t - 8) ** 2
    # one_dim_search(ft, t)
    f_r_method(fx, x1, x2, {'x1': 0, 'x2':0})
    
