import matplotlib.pyplot as plt
import numpy as np

import numpy as np
def gbar(K,N,model,domain,codomain):
    """     Parameters:
        K = number of datasets to produce
        N = size of each dataset
        model = the model to evaluate
        domain = the domain of the target f (1-D array)
        codomain = the values of the target f (1-D array)
            Output:
        G is a 2-D numpy array.
        The rows of G are g^D for various datasets D.
        The columns of G are the x in domain(f)
        An entry D,x is g^D(x).
        The mean of G along the 0 axis is gbar
    """

    G=[]

    for i in range(K):
        X=[]
        y=[]
        for n in range(N):
            index = np.random.randint(0,len(domain))
            x = domain[index]
            yy = codomain[index]
            X.append(x)
            y.append(yy)
        X = np.array(X)
        y = np.array(y)
        model.fit(X,y)
        g = model.predict(domain)
        G.append(g)
    return np.array(G)

def variance(G):
    return np.mean((G-np.mean(G,axis=0))**2)

def bias(G,codomain):
    return np.mean((np.mean(G,axis=0)-codomain)**2)





def RMSE(y,yhat):
    return np.sqrt(np.sum((yhat-y)**2)/len(y))


def add_bias(df):
    """Add bias column to pandas dataframe"""
    df["bias"] = np.ones(df.shape[0])
    col2=[df.columns[-1]]+list(df.columns[:-1])
    df= df[col2]
    return df

def noisycurve(f,dom,noise):
    y =f(dom)+noise
    plt.scatter(dom,y,alpha=0.5)
    return y

def getH(X):
    return X.dot(np.linalg.pinv(X))

def R2(y,yhat):
    SSres = np.sum((y-yhat)**2)
    mean = np.mean(y)
    SStot = np.sum((y-mean)**2)
    return 1-SSres/SStot


def make_nonlinear(N=50,sig=2):
    """
    def make_nonlinear(N=30,sig=1):
    return X,y
    """
    mu = [0,0]
    X=np.ones(3*N).reshape(N,3)
    X[:,0] = np.ones(N)
    X[:,1]=np.random.randn(N)*sig+mu[0]
    X[:,2]=np.random.randn(N)*sig+mu[1]

    xx,yy = X[:,1],X[:,2]
    xmax = 1.1*np.max(xx)
    xmin = 1.1*np.min(xx)
    ymax = 1.1*np.max(yy)
    ymin = 1.1*np.min(yy)

    y = (xx**2+yy**2 < sig**2)*2-1
    Xg = X[y==1]
    Xb = X[y==-1]

    plt.scatter(Xg[:,1],Xg[:,2],c="b")
    plt.scatter(Xb[:,1],Xb[:,2],c='r')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    x1 = np.arange(xmin,xmax,0.01)
    x2 = np.arange(ymin,ymax,0.01)

    x1v,x2v = np.meshgrid(x1,x2)

    z = np.sign(x1v**2+x2v**2 - sig**2)


    plt.contourf(x1v,x2v,z,alpha=0.25)

    plt.title("Inherently nonlinearly separable")
    plt.show()

    return X,y


def myblobs(N=30,sig_yes=1,sig_no=1):
    """
    def myblobs(N=30,sig_yes=1,sig_no=1):
    return X,y,w
    """
    mu_yes = [-2,0]
    X_yes=np.ones(3*N).reshape(N,3)
    X_yes[:,0] = np.ones(N)
    X_yes[:,1]=np.random.randn(N)*sig_yes+mu_yes[0]
    X_yes[:,2]=np.random.randn(N)*sig_yes+mu_yes[1]

    y_yes=np.ones(N)

    mu_no = [3,0]
    X_no=np.ones(3*N).reshape(N,3)
    X_no[:,0] = np.ones(N)
    X_no[:,1]=np.random.randn(N)*sig_no+mu_no[0]
    X_no[:,2]=np.random.randn(N)*sig_no+mu_no[1]

    y_no=np.ones(N)*(-1)

    X = np.vstack((X_yes,X_no))
    y = np.hstack((y_yes,y_no))
    w = np.array([3,-2,-2.5])
    return X,y,w

def lin_boundary(w,X,y):
    """
    lin_boundary(w,X,y,xmin=-8,xmax=8,ymin=-8,ymax=8)
    """
    xmin = np.min(X[:,1])-0.5
    xmax = np.max(X[:,1])+0.5
    ymin = np.min(X[:,2])-0.5
    ymax = np.max(X[:,2])+0.5
    x1 = np.arange(xmin,xmax,0.01)
    x2 = np.arange(ymin,ymax,0.01)

    x1v,x2v = np.meshgrid(x1,x2)
    a,b,c = w[0],w[1],w[2]

    z = np.sign(a+b*x1v+c*x2v)
    plt.contourf(x1v,x2v,z,alpha=0.25)
    Xg = X[y==1]
    Xb = X[y==-1]

    plt.scatter(Xg[:,1],Xg[:,2],c="b",alpha=0.5)
    plt.scatter(Xb[:,1],Xb[:,2],c='r',alpha=0.5)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    plt.title("A classification boundary")
    plt.show()


def compare_boundary(w,ww,X,y,xmin=-8,xmax=8,ymin=-8,ymax=8):
    """
    lin_boundary(w,X,y,xmin=-8,xmax=8,ymin=-8,ymax=8)
    """
    x1 = np.arange(xmin,xmax,0.01)
    x2 = np.arange(ymin,ymax,0.01)

    x1v,x2v = np.meshgrid(x1,x2)
    a,b,c = w[0],w[1],w[2]
    z = np.sign(a+b*x1v+c*x2v)
    plt.contourf(x1v,x2v,z,alpha=0.25)
    a,b,c = ww[0],ww[1],ww[2]
    z = np.sign(a+b*x1v+c*x2v)
    plt.contourf(x1v,x2v,z,alpha=0.25)
    Xg = X[y==1]
    Xb = X[y==-1]

    plt.scatter(Xg[:,1],Xg[:,2],c="b")
    plt.scatter(Xb[:,1],Xb[:,2],c='r')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    plt.title("A classification boundary")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit



def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

