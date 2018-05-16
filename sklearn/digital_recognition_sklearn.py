import numpy as np
import random
import pickle
from sklearn.neural_network import MLPClassifier as MLPC

training_data = pickle.load(open("train.p", "rb"))
test_data = pickle.load(open("test.p", "rb"))

training_data_size = len(training_data)
total_vector = []

def five_folds(data, remainder):
    training_feature = []
    training_label = []
    validation_feature = []
    validation_label = []
    for i in range(len(data)):
        if i % 5 == remainder:
            validation_feature.append(data[i][0])
            validation_label.append(data[i][1])
        else:
            training_feature.append(data[i][0])
            training_label.append(data[i][1])
    return np.array(training_feature), np.array(training_label), np.array(validation_feature), np.array(validation_label)

""" Combine all of the parameters"""
alpha = [1, 0.1, 0.01, 0.001, 0.0001]
activation_function = ['logistic', 'tanh', 'relu']
hidden_layer_sizes = [(50, 50), (100, 100), (20, 20, 20), (50, 50, 50), (100, 100, 100)]
max_iteration = [100, 150, 200, 250, 300]

parameter_set = []
for i in alpha:
    for j in activation_function:
        for k in hidden_layer_sizes:
            for l in max_iteration:
                parameter_set.append([i, j, k, l])

for current_para in parameter_set:
    best_classifier = None
    highest_accuracy = 0
    
    """ get current parameters for the classifier """
    current_alpha = current_para[0]
    current_activition_function = current_para[1]
    current_hidden_layer = current_para[2]
    current_iteration = max_iteration[3]
    
    fold_num = 0
    for i in range(0, 5):
        fold_num = fold_num + 1
        """ build up a classifier """
        current_classifier = MLPC(hidden_layer_sizes = current_hidden_layer, activation = current_activition_function, alpha = current_alpha, max_iter = current_iteration)
        """ get training and validation set """
        current_training_feature, current_training_label, current_validation_feature, current_validation_label = five_folds(training_data, i)

        """ training """
        current_classifier.fit(current_training_feature, current_training_label)

        """ validation """
        correct_num = 0
        number_of_instance_in_validation_set = len(current_validation_feature)

        for i in range(number_of_instance_in_validation_set):
            if current_classifier.predict(current_validation_feature[i].reshape(1, -1)) == current_validation_label[i]:
                correct_num = correct_num + 1

        print ("     Correct number for this time is ", correct_num)
        correct_percentage = float(correct_num)/(number_of_instance_in_validation_set)
        print ("     Accuracy for this time is ", correct_percentage)

        if correct_percentage > highest_accuracy:
            highest_accuracy = correct_percentage
            best_classifier = current_classifier

# Test
    correct_num_on_test = 0
    number_of_instance_in_test_data = len(test_data)
    for test_instance in test_data:
        if  best_classifier.predict(np.array(test_instance[0]).reshape(1, -1))[0] == test_instance[1]:
            correct_num_on_test = correct_num_on_test + 1
    correct_percentage_on_test = float(correct_num_on_test) / (number_of_instance_in_test_data)
    print ("For:")
    print ("     current learning rate: ", current_para[0])
    print ("     current activition function: ", current_para[1])
    print ("     current hidden layer size: ", current_para[2])
    print ("     current max iteration: ", current_para[3])
    print ("     Accuracy is ", correct_percentage_on_test * 100, "%")
    print ("                 ")
    print ("                 ")

    total_vector.append(correct_percentage_on_test)

best_para_index = np.argmax(total_vector)
print ("The best parameter combo is: ")
print ("     Learning Rate: ", parameter_set[best_para_index][0])
print ("     Activition function: ", parameter_set[best_para_index][1])
print ("     Hiddent layer size: ", parameter_set[best_para_index][2])
print ("     Max iteration: ", parameter_set[best_para_index][3])
print ("     Accuracy is: ", total_vector[best_para_index] * 100, "%")


