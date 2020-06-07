# Your grade

A-

# Instructor comments

I'm psyched you tried this!  You're the only person who did. I'm pasting a solution below .  I changed things a little so that it's easy to swap out activation functions.

```
import numpy as np
## ELU
alpha = 10**0
theta = lambda ss: (ss>0)*ss + (ss <= 0)*alpha*(np.exp(ss)-1)
thetaprime = lambda ss: (ss>0)*1 + (ss <= 0)*alpha*np.exp(ss)
    
def init_nn(D_l,sigma=1):
    """
    Initialize Artificial Neural Network
    Input:  Numpy array D_l is a list of layer sizes from 0 to L of an ANN
    Output: A python list of weight matrixes W(l) for l = 1..L (inclusive)
    The weight matrixes are numpy 2d arrays and are initialized from the standard
    normal distribution.
    """
    #return x_init_nn(D_l,sigma)
    np.random.seed(22)
    L = D_l.shape[0]-1
    w = []
    for l in range(1,L+1):
        d_prev = D_l[l-1]
        d_l = D_l[l]
        W_l = np.random.randn((d_prev+1)*d_l).reshape((d_prev+1),d_l)
        W_l = W_l*sigma #we want _small_ initial weights: sigma*max(norm(x)) << 1
        w.append(W_l)
    return w


def x_init_nn(D_l,sigma):
    np.random.seed(22)
    L = D_l.shape[0]-1
    w = []
    for l in range(1,L+1):
        d_prev = D_l[l-1]
        d_l = D_l[l]
        sigma = 2*np.sqrt(1/(d_prev+1+d_l))
        W_l = np.random.normal(0,sigma**2,(d_prev+1)*d_l).reshape((d_prev+1),d_l)
        
        #W_l = W_l*sigma #we want _small_ initial weights: sigma*max(norm(x)) << 1
        w.append(W_l)
    return w

    
def forward_prop(x,w,regression = False):
    """
    Input: 
        d+1 dimensional input vector x to the neural network;
        length L-1 list w of weight matrixes
        optional threshold (activation) function theta
    Output:
        An array xx of intermediate layer output values, l=0,...,L
        An array s of intermediate layer input values, l=1,...,L        
    To be consistent with the indexing from the book, s begins with a
    placeholder 0.  Please ignore this value.  (Notice that s is never
    actually used in the code, and in an optimized implementation we 
    would not even compute it directly.)
    """
    assert(x[0]==1) #x should have the bias node
    assert(len(x) == w[0].shape[0] ) #make sure the sizes match
    
    xx = []
    s = [0]  # initial zero used only to simplify indexing
    xx.append(x)
    l=1
    for W in w[:-1]:
        s.append(W.T.dot(xx[l-1]))
        xx.append(np.hstack((np.array([1]),theta(s[l]))))
        l += 1
        
    W = w[-1]
    s.append(W.T.dot(xx[l-1]))

    if regression:
        xx.append(s[l])
    else: #classification
        xx.append(theta(s[l]))
    return xx,s

def back_prop(s,xx,w,yi,regression=False):
    assert(np.isscalar(yi)) # yi is just the y value for the instance x
    delta = []
    L = len(w)
    if regression:
        delta.append(2*(s[L]-yi)) #delta_L, assume theta = id
    else: #classification
        delta.append(2*(theta(s[L])-yi)*thetaprime(s[L])) #delta_L, assume theta = relu
    l = L-1
    for W in reversed(w[1:]):
        delta_lp1 = delta[-1]
        delta_l = thetaprime(s[l])*(W.dot(delta_lp1)[1:])
        delta.append(delta_l)
        l -= 1
    return list(reversed(delta))

def nn_gradient(xx,delta):
    grad = []
    L = len(xx)-1    
    for x,d in zip(xx[:-1],delta):
        _x = x.reshape(x.shape[0],1)
        _d = d.reshape(d.shape[0],1)
        grad.append(_x.dot(_d.T))
    return grad

def nn_stoch_grad_des(X,y,D_l,eta=0.01,iterations=50,regression = False,sigma=1,x_init=False):
    np.random.seed(100)
    if x_init:
        w = x_init_nn(D_l,sigma)
    else:
        w = init_nn(D_l,sigma)
    for i in range(iterations):
        shuff = np.random.permutation(X.shape[0])
        Xs = X[shuff]
        ys = y[shuff]
        for x,yi in zip(Xs,ys):
            xx,s = forward_prop(x,w,regression)
            delta = back_prop(s,xx,w,yi,regression)
            grad = nn_gradient(xx,delta)
            new_w = []
            for wi,w_g in zip(w,grad):
                new_w.append(wi - eta*w_g)
            w = new_w
    return w

def nn_predict(x,w,regression=False):
    xx,s = forward_prop(x,w,regression)
    if not regression: #classification
        return np.sign(xx[-1])
    else:
        return xx[-1]
        
        
def nn_predict_all(X,w,regression=False):
    g = np.ones(X.shape[0])
    for i,x in enumerate(X):
        g[i] = nn_predict(x,w,regression)
    return g


````
