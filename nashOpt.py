# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:34:54 2021

@author: p4u1
"""

import numpy as np
import scipy.optimize as opt

def primalLinearProgOpt(A, n, meth = 'simplex'):
    """ Wrapper function to get primal optimization results
        Uses scipy.optimize.linprog method
    
    Arguments:  A - square n x n payoff matrix (must be ndarray)
                n - size of matrix A
                method - linear programming method to use
                
    Returns:    Optimization variable values
                Objective function value"""

    # Objective function coefficents
    c0 = [0 for i in range(n)]
    c0.append(-1)
    c0 = np.array(c0)
    
    # Inequality contraints
    a = np.ones((n,1))
    A_u = np.concatenate((-(A.T), a), axis = 1)
    b_u = np.zeros(n)
    
    # Equality contraints
    A_e = [1 for i in range(n)]
    A_e.append(0)
    A_e = np.array(A_e)
    A_e = A_e.reshape((1,n+1))
    b_e = np.array(1)
    
    # Bounds
    bound = [(0, None) for i in range(n)]
    bound.append((None,None))
    
    # Run scipy.optimize.linprog method
    results = opt.linprog(c=c0,
                          A_ub=A_u, b_ub=b_u,
                          A_eq=A_e, b_eq=b_e,
                          bounds=bound, 
                          method=meth)
    
    return results.x

def dualLinearProgOpt(A, n, meth = 'simplex'):
    """ Wrapper function to get dual optimization results
        Uses scipy.optimize.linprog method
    
    Arguments:  A - square n x n payoff matrix (must be ndarray)
                n - size of matrix A
                method - linear programming method to use
                
    Returns:    Optimization variable values"""

    # Objective function coefficents
    c0 = [0 for i in range(n)]
    c0.append(1)
    c0 = np.array(c0)
    
    # Inequality contraints
    a = np.ones((n,1))
    A_u = np.concatenate((A, -a), axis = 1)
    b_u = np.zeros(n)
    
    # Equality contraints
    A_e = [1 for i in range(n)]
    A_e.append(0)
    A_e = np.array(A_e)
    A_e = A_e.reshape((1,n+1))
    b_e = np.array(1)
    
    # Bounds
    bound = [(0, None) for i in range(n)]
    bound.append((None,None))
    
    # Run scipy.optimize.linprog method
    results = opt.linprog(c=c0,
                          A_ub=A_u, b_ub=b_u,
                          A_eq=A_e, b_eq=b_e,
                          bounds=bound, 
                          method=meth)
    
    return results.x

# Test area
# A = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
# n = 3

# resultsPrimal = primalLinearProgOpt(A, n, 'interior-point')
# resultsDual = dualLinearProgOpt(A, n)
    