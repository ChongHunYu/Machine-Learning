Great job on number 1.

Very good job on number 2.

Thank you for making it so easy for me to see what's happening in this problem and number 3.

Number 3 looks good.

For number 4 I think there's something wrong.  I ran it and clobbered the output you had.

But $y$ isn't recognized because it hasn't been declared before:

```
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-2-496cf72b8c47> in <module>()
      7 #Q1
      8 x=np.arange(-5,6)
----> 9 plt.plot(y)
     10 #Q2
     11 theta=np.arange(0,2*np.pi,1./100)

NameError: name 'y' is not defined
```

This is a bad thing about python notebooks, that you can run cells out of order so that the code runs, but then when the cells are run in order the code breaks.  

For Q1 it should be

```
plt.plot(x,x**2)
```

and your domain needs more points. 

Q2 looks right.

For Q3 you should look at the contour plot syntax...

[here](https://matplotlib.org/3.1.3/gallery/images_contours_and_fields/contour_demo.html#sphx-glr-gallery-images-contours-and-fields-contour-demo-py)

The last problem looks good.

Excellent work.


