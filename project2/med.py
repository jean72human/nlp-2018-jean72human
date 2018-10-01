
# coding: utf-8

import sys


import numpy as np

def costs(l1='!nspb',l2='!nspb'):
    """
    Gives the deletion, insertion, and substitution cost for a pair of letters
    """
    return (1,1,0) if l1 == l2 else (1,1,2)
        

def med(str1, str2):
    """
    Takes two strings as input and outputs their minimum edit distance
    """
    
    m = len(str1)
    n = len(str2)
    
    
    ## costs
    d, i, s = costs()
    
    D = np.zeros((m+1,n+1), np.int64)
    
    
    ## intitialize array with base case
    for k in range(1,m+1):
        D[k,0] = D[k-1,0] + d
        
    for j in range(1,n+1):
        D[0,j] = D[0,j-1] + i
        
    ## compute MEDistance
    for k in range(1,m+1):
        for j in range(1,n+1):
            d,i,s = costs(str1[k-1],str2[j-1])
            D[k,j] = min((D[k-1,j]+d),
                         (D[k-1,j-1]+s),
                         (D[k,j-1]+i))
            
    print ("The minimum edit distance between",str1," and ",str2," is: ",D[m,n])
    




med(sys.argv[1],sys.argv[2])

