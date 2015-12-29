###############################################################################
# Author:         Alan Ramponi                                                #
# Date:           October 15th, 2015                                          #
# Description:    Naive OCR (oooh fly down, I'm just getting started!)        #
###############################################################################

from sklearn import datasets, svm
import pylab as pl


# load the digits dataset
digits_dataset = datasets.load_digits()

# Each datapoint is a 8x8 image of a digit. We have:
# Samples (rows):       1797 (~180 samples for each class)
# Features (cols):        64 (with int values from 0. to 16.)
# Classes:                10 (all the digits: 0,1,2,3,4,5,6,7,8,9)

# initialize an SVC estimator for the classification task
classifier = svm.SVC(gamma=0.001, C=100.)   # black box parameters!

# fit the estimator to the model (i.e. it learns from the model)
# training set:         all the images apart from the first
classifier.fit(digits_dataset.data[0:], digits_dataset.target[0:])

# predict the class of the first image (not used to train the classifier)
predicted_class = classifier.predict(digits_dataset.data[0])

# plot the original image to check visually the number in it
pl.gray()
pl.matshow(digits_dataset.images[0])
pl.show()

# print the number that my beautiful classifier predicts
print predicted_class
