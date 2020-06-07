import matplotlib.pyplot as plt
import numpy as np

def myblobs(N=30,sig_yes=1,sig_no=1):
    #np.random.seed(11)
    mu_yes = [-2,0]
    X_yes=np.ones(3*N).reshape(N,3)
    X_yes[:,0] = np.ones(N)
    X_yes[:,1]=np.random.randn(N)*sig_yes+mu_yes[0]
    X_yes[:,2]=np.random.randn(N)*sig_yes+mu_yes[1]

    y_yes=np.ones(N)

    mu_no = [3,2]
    X_no=np.ones(3*N).reshape(N,3)
    X_no[:,0] = np.ones(N)
    X_no[:,1]=np.random.randn(N)*sig_no+mu_no[0]
    X_no[:,2]=np.random.randn(N)*sig_no+mu_no[1]

    y_no=np.ones(N)*(-1)

    X = np.vstack((X_yes,X_no))
    y = np.hstack((y_yes,y_no))
    w = np.array([3,-2,-2.5])
    return X,y,w

def lin_boundary(w,X,y,xmin=-8,xmax=8,ymin=-8,ymax=8):
    """
    lin_boundary(w,X,y,xmin=-8,xmax=8,ymin=-8,ymax=8)
    """
    x1 = np.arange(xmin,xmax,0.01)
    x2 = np.arange(ymin,ymax,0.01)

    x1v,x2v = np.meshgrid(x1,x2)
    a,b,c = w[0],w[1],w[2] 

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

import matplotlib.pyplot as plt
import numpy as np

def myblobs(N=30,sig_yes=1,sig_no=1):
    #np.random.seed(11)
    mu_yes = [-2,0]
    X_yes=np.ones(3*N).reshape(N,3)
    X_yes[:,0] = np.ones(N)
    X_yes[:,1]=np.random.randn(N)*sig_yes+mu_yes[0]
    X_yes[:,2]=np.random.randn(N)*sig_yes+mu_yes[1]

    y_yes=np.ones(N)

    mu_no = [3,2]
    X_no=np.ones(3*N).reshape(N,3)
    X_no[:,0] = np.ones(N)
    X_no[:,1]=np.random.randn(N)*sig_no+mu_no[0]
    X_no[:,2]=np.random.randn(N)*sig_no+mu_no[1]

    y_no=np.ones(N)*(-1)

    X = np.vstack((X_yes,X_no))
    y = np.hstack((y_yes,y_no))
    w = np.array([3,-2,-2.5])
    return X,y,w

def lin_boundary(w,X,y,xmin=-8,xmax=8,ymin=-8,ymax=8):
    """
    lin_boundary(w,X,y,xmin=-8,xmax=8,ymin=-8,ymax=8)
    """
    x1 = np.arange(xmin,xmax,0.01)
    x2 = np.arange(ymin,ymax,0.01)

    x1v,x2v = np.meshgrid(x1,x2)
    a,b,c = w[0],w[1],w[2] 

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

