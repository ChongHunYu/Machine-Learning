Hi Chong,

This PLA is perfect.

Here are my solutions (because you asked in class):

	def PLA(X,y,w,max_iter=1000):
		for i in range(max_iter):
			mc = (np.sign(X.dot(w)) != y)
			if not mc.any():
				break
			badx = X[mc][0]
			bady = y[mc][0]
			w = w+bady*badx
		return w,i

and pocket:

	def Pocket(X,y,w,max_iter=1000):
		E_in_best = 1
		E_in_argbest = w
		for i in range(max_iter):
			mc = (np.sign(X.dot(w)) != y)
			if not mc.any():
				break
			if E_in(X,y,w) < E_in_best:
				E_in_best = E_in(X,y,w)
				E_in_argbest = w
				
			badx = X[mc][0]
			bady = y[mc][0]
			w = w+bady*badx
		return E_in_argbest,i

and scaling:

	def meannorm(X):
		mins = np.min(X,axis=0)
		maxs = np.max(X,axis=0)
		mu = np.mean(X,axis=0)
		bias = np.allclose(X[:,0],np.ones(X.shape[0]))  ## Is there a bias column?
		if bias:  ## Without this max=min, causes divide-by-zero
			maxs[0]=1
			mins[0]=0
			mu[0]=0
		return mins,maxs,mu

	def meannorm_apply(X,mins,maxs,mu):
		return (X-mu)/(maxs-mins)
		
		
Your pocket is great.

With scaling there are some issues.  It looks like you are trying to get the means and apply the transform at the same time.  But we want to keep getting the parameters separate from applying the transform.  Because we want to apply the transform *to the test set* but only with the parameters that come from the training set.  I'm not sure why I used italics in the last sentence.  

Overall this is very good work.
		
