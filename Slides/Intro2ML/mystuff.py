import matplotlib.pyplot as plt
import numpy as np

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

