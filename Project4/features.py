# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples
import matplotlib.pylab as plt # Yantian
from threading import Thread # Yantian
import time

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.
    colorbar()
    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
    Idea based on feature1.py's implementation of white contiguous spaces 
    My implementation takes in individual pixels and checks if they correspond to a white pixel and have not been previously visited, for each 
    pixel there has to be a contiguous white space therefore I keep a count, the pixel is passed to the graph search function which adds it to the 
    closed list and recursively calls the graph search function on its neighbours. 
    The enhanced features takes into account the contiguous white spaces of the different digits 
    The function returns [1, 0, 0] when it encounters one contiguous white space
                         [0, 1, 0] when it encounters two contiguous white spaces
                         [0, 0, 1] when it encounters three or more white spaces 
    Omitting certain neighbours [pixel values], I am able to achieve a test accuracy of 89.7% 

    ##
    """
    features = basicFeatureExtractor(datum)

    "*** YOUR CODE HERE ***"           
    N = 28
    new_features = np.zeros(3, int)
    closedSet = set()
    count = 0 
    fringe = util.Queue()

    def graphSearch(datum, width, height, closedSet):
        if 0 <= width < len(datum) and 0 <= height < len(datum) and datum[width][height] == 0:
            closedSet.add((width, height))
            for i, j in [(width-1, height), (width-1, height+1), (width, height-1), (width, height+1), (width+1, height)]:
                if (i, j) not in closedSet: 
                    graphSearch(datum, i, j, closedSet)

    for width in range(N):
        for height in range(N):
            if datum[width][height] == 0 and (width, height) not in closedSet:
                count += 1
                graphSearch(datum, width, height, closedSet)
            
    if count == 1:
        new_features = [1, 0, 0]

    if count == 2:
        new_features = [0, 1, 0]

    if count >= 3:
        new_features = [0, 0, 1]

    return np.append(features, new_features)

def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit
    
    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
