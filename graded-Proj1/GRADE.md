# Your grade

A

# Instructor comments

Hi Chong,

With proj1 I think we already mentioned that 

    return (-yy * x)/(1+np.exp(yy*w.T*x))

should be

    return (-yy * x)/(1+np.exp(yy*w.T.dot(x)))

in the logistic regression gradient.

After this small change your error goes down to 2%, actually outperforming the library LogisticRegression.

(I also reduced your eta to 0.01)


It looks like your Part 2 works perfectly.  Nice job. 


 