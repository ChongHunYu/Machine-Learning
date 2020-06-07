def softboundary():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_blobs

    X,y = make_blobs(centers=2,cluster_std=3.0)
    clf = LogisticRegression()
    clf.fit(X,y)
    w0,w1,w2 = clf.intercept_[0],clf.coef_[0,0],clf.coef_[0,1]
    w0,w1,w2

    import matplotlib.pyplot as plt
    import numpy as np
    yes = y==1
    no = ~yes


    xmin = np.min(X,axis=0)[0]*1.1
    xmax = np.max(X,axis=0)[0]*1.1
    ymin = np.min(X,axis=0)[1]*1.1
    ymax = np.max(X,axis=0)[1]*1.1

    xx = np.linspace(xmin,xmax)
    line = (-w1*xx-w0)/w2
    plt.scatter(X[yes][:,0],X[yes][:,1])
    plt.scatter(X[no][:,0],X[no][:,1])
    #plt.plot(xx,line)




    xx = np.linspace(xmin, xmax, 100)
    yy = np.linspace(ymin, ymax, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    probabilities = clf.predict_proba(Xfull)
    imshow_handle = plt.imshow(probabilities[:, 1].reshape((100, 100)),
                                       extent=(xmin,xmax,ymin,ymax), origin='lower')

    plt.title("Probabilities")

    ax = plt.axes([0.15,-.15, 0.7, 0.05])
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

    return plt
