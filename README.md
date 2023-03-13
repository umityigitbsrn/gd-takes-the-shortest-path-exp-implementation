# Overparameterized Nonlinear Learning: Gradient Descent Takes the Shortest Path?
Implementation of the experiment mentioned in "Overparameterized Nonlinear Learning: Gradient Descent Takes
the Shortest Path?" paper by Samet Oymak and Mahdi Soltanolkotabi using PyTorch

## MNIST Experiment
Implementation of this experiment is under 'mnist_experiment' folder

LeNet model is used to conduct this experiment. This model has two convolutional layers followed by two fully-connected layers. 
Instead of cross-entropy loss, least-squares loss is used, without softmax layer,
which falls within nonlinear least-squares framework. They conducted two set of
experiments with n = 500 and n = 5000. Both experiments use Adam with learning rate
0.001 and batch size 100 for 1000 iterations. At each iteration, they record the
normalized misfit and distance to obtain a misfit-distance trajectory similar. 
They repeat the training 20 times (with independent initialization 
and dataset selection) to obtain the typical behavior.

## Low-rank Regression
Implementation of this experiment is under 'low_rank_regression' folder

Gradient for the loss function of low rank regression can be mentioned in the paper on page 32 (in Appendix).
The implementation will be based on this notation

The experiment setup is taken from the page 13 (under 6.2 Low-rank Regression) 