# scyfer challenge
Run through the Theano Deep Learning example:
http://deeplearning.net/tutorial/lenet.html#lenet

Add the following functionality to the example:
Adadelta learning rate update
Dropout 
PreLU units

You have to implement this based on your own insight. Copying code from other places is not allowed.
Run short experiments to see if the added functionalities add anything to the performance of the system.

# Results
The experiments were performed in jupyter notebook and the results can be found <a href="https://github.com/daviddemeij/scyfer-challenge/blob/master/scyfer-challenge.ipynb">here</a>.

# Code
The added functionalities are coded in the <a href="https://github.com/daviddemeij/scyfer-challenge/blob/master/convolutional_mlp.py">convolution_mlp.py file</a>, I transformed the initial evaluate_lenet5() function into a python class for easier experimentation.
