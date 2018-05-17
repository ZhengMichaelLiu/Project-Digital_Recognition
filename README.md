# Project-Digital_Recognition

trainingimages: 5000 training digits, around 500 samples from each digit class. Each digit is of size 28x28, and the digits are concatenated together vertically. The file is in ASCII format, with three possible characters. ' ' means a white (background) pixel, '+' means a gray pixel, and '#' means a black (foreground) pixel. 

traininglabels: a vector of ground truth labels for every digit from trainingimages.

testimages: 1000 test digits (around 100 from each class), encoded in the same format as the training digits.

testlabels: ground truth labels for testimages.

With sklearn library, done

The best parameter combo is:

    Learning Rate: 0.0001
    
    Activition function: relu
    
    Hidden layer size: (100, 100, 100)
    
    Max iteration: 250
    
    Accuracy is: 91.7%
    

With tensorflow ------
