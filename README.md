# A-Two-Step-Disentanglement-Method
Model implementation and trained network weights for "A Two-Step Disentanglement Method" by Naama Hadad, Lior Wolf and Moni Shahar from Tel Aviv University.
This work address the problem of disentanglement of factors that generate a given data into those that are correlated with the labeling and those that are not. The model employs adversarial training in a straightforward manner.

# Requirements
This code runs using Python 2.7, numpy, Keras and Theano. It also uses matplotlib to plot results.
The code was developed on Ubuntu.

# Usage
In current version we provide model + weights for the Sprites dataset as well as small part of the test set to demonstrate results.
We are using a preprocessed version of this dataset.

To run the code, download the python files as well as the model and data files and use the following syntax:  
`python test_net.py`

To generate additional examples of the disentanglement over the partial data you can rerun the code to randomize different characters. This code can also be used over the full Sprites dataset after applying similar resolution preprocessing.
