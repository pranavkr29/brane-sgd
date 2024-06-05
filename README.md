# String Theory and Machine Learning Loss Landscape
## TF and Pytorch Code to find the minima in string brane configurations using SGD

### How to find all the minima of the landscapes predicted by different brane configuration, using ML optimization methods

A code repository for papers listed below

https://link.springer.com/article/10.1007/JHEP10(2023)107

https://arxiv.org/abs/2312.04643

## Workflow

All the codefiles in this repository are some permutation of an algorithm where the potential function created using physical principles is cast as a landscape, whereupon we start from a randomly chosen point, and use SGD(or any of the other Keras optmizer) to find a minima, using the value of the loss function as a threshold to stop the run. Because the number of minima changes with a Hyperparameter of the cost function (N or N4 in the codes), the code needs to run many many times, sometimes as many as 1000 times, hence the code is paralellized and we have used @tf.function to speed it up.  

## Structure

All the .py files are self contained files. There is no directory structure as we were running them on colab or Kaggle often, and keeping track of all the changes over a directory was cumbersone, so we stuck one script which needed to be edited to change the code. 


## Getting started

To get started, first install the required libraries inside a virtual environment:

`pip install -r requirements.txt`
