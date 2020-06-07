### Project 1

In this project you will implement the entire pipeline for applying a linear model to both a classification and a regression problem.

#### Part 1 Classification

Step 1:  Find a binary classification dataset.  Good places to look:  Kaggle, OpenML, UCI Machine Learning Repository.

Suggestion:  You will simplify your life if you select data that is at least mostly numerical rather than nominal.  Nominal data will have to be numerically encoded to apply a linear model. 

Step 2:  Load the data using pandas.  Convert any nominal columns to numerical columns. This is also a convenient place to convert target values to 1 or -1. 

Step 3:  Convert the pandas dataframe to a numpy X and y.  Make sure that X has a bias column and that y contains only values of $\pm 1$.

Step 4:  Perform a train-test split on the data.  You can use your own code or sklearn for this.

Step 5:  Scale the data.  You can use your own code or sklearn for this.  Be sure you scale correctly:  First fit the training data to get the scaling parameters, and then apply the scaling to both the training and testing data.

Step 6:  Train a linear classifier such as logistic regression on the training data. This must be your own code.

Step 7:  Apply your model to the test data to get predictions $\hat{y}$.

Step 8:  Evaluate the accuracy of your model.  You can use simply the proportion of correct predictions.

Step 9: (Optional) Try doing some transformations of features to improve accuracy.  After you add or modify features (on the whole dataset) start again at step 4.

#### Part 2 Regression

Like the above, but use linear regression rather than logistic regression.

You will need a new dataset in which the targets are real valued.

In this case you do **not** want to convert y to 1 and -1.

Please use the $R^2$ error metric.


