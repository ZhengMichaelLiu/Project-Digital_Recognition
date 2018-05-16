import numpy as np
import pickle
import random

# read in data
training_data = pickle.load(open("train.p", "rb"))
test_data = pickle.load(open("test.p", "rb"))

# create validation data
fold0 = []
fold1 = []
fold2 = []
fold3 = []
fold4 = []

for i in range(0, len(training_data)):
    if i % 5 == 0:
        fold0.append(training_data[i])
    elif i % 5 == 1:
        fold1.append(training_data[i])
    elif i % 5 == 2:
        fold2.append(training_data[i])
    elif i % 5 == 3:
        fold3.append(training_data[i])
    elif i % 5 == 4:
        fold4.append(training_data[i])

train1 = fold1 + fold2 + fold3 + fold4
valid1 = fold0

train2 = fold0 + fold2 + fold3 + fold4
valid2 = fold1

train3 = fold0 + fold1 + fold3 + fold4
valid3 = fold2

train4 = fold0 + fold1 + fold2 + fold4
valid4 = fold3

train5 = fold0 + fold1 + fold2 + fold3
valid5 = fold4

train_validation_pair = [[train1, valid1], [train2, valid2], [train3, valid3], [train4, valid4], [train5, valid5]]

# set parameters
learning_rate = [0.05, 0.01, 0.005, 0.001, 0.0005]
learning_rate_decay = ['yes', 'no']
data_order = ['random', 'fixed']
initial_weight = ['zeros', 'random']
epoch_num = [50, 80, 100, 150, 200]

parameter_list = []
for each_learning_rate in learning_rate:
    for each_learning_rate_decay in learning_rate_decay:
        for each_data_order in data_order:
            for each_initial_weight in initial_weight:
                for each_epoch_num in epoch_num:
                    parameter_list.append([each_learning_rate, each_learning_rate_decay, each_data_order, each_initial_weight, each_epoch_num])

total_vector = []

for each_parameter_comb in parameter_list:
    # Train
    # current learning rate, same for all folds
    current_learning_rate = each_parameter_comb[0]
    fold_num = 0

    # add up correct percent after each pair
    total_correct_percent = 0.000
    # add up weights after each pair
    total_weights = []
    for i in range(0, 10):
        total_weights.append(np.zeros(785))
    
    # for each fold, do train and validation
    for each_pair in train_validation_pair:
        training_set = each_pair[0]
        validation_set = each_pair[1]
        fold_num = fold_num + 1
        # initial weights
        weight_vector_for_ten_perceptron = []
        # zeros intial weights
        if each_parameter_comb[3] == 'zeros':
            for i in range(0, 10):
                # with one bias term
                weight_vector_for_ten_perceptron.append(np.zeros(785))
        # random set intial weights
        elif each_parameter_comb[3] == 'random':
            for i in range(0, 10):
                weight_vector_for_ten_perceptron.append(np.zeros(785))
                for j in range(0, 785):
                    weight_vector_for_ten_perceptron[i][j] = np.random.normal(0, 0.3)
        
        # epoch
        for epoch_index in range(1, each_parameter_comb[4] + 1):
            print("This is ", epoch_index, " times in fold ", fold_num)
            # change learning rate
            if each_parameter_comb[1] == 'yes':
                current_learning_rate = current_learning_rate / epoch_index
            elif each_parameter_comb[1] == 'no':
                current_learning_rate = current_learning_rate

            if each_parameter_comb[2] == 'random_order':
                random.shuffle(training_set)
            
            for i in range(0, 10):
                for each_data in training_set:
                    # attributes and bias term [1]
                    current_predict = np.sign(np.dot(weight_vector_for_ten_perceptron[i], each_data[0] + [1]))
                    if each_data[1] == i:
                        shouldbe = 1
                    else:
                        shouldbe = -1

                    if current_predict != shouldbe:
                        # update the weights
                        weight_vector_for_ten_perceptron[i] = np.add(weight_vector_for_ten_perceptron[i], current_learning_rate * shouldbe * np.array(each_data[0] + [1]))
        
        # validation
        correct = 0
        for each_valid in validation_set:
            prediction_ten_perceptron = []
            for i in range(10):
                current_result = np.dot(weight_vector_for_ten_perceptron[i], each_valid[0] + [1])
                prediction_ten_perceptron.append(current_result)
            maybe_correct_perceptron = np.argmax(prediction_ten_perceptron)
            if maybe_correct_perceptron == each_valid[1]:
                correct = correct + 1
        correct_percentage = float(correct / len(validation_set))

        total_correct_percent = total_correct_percent + correct_percentage
        for i in range(0, 10):
            total_weights[i] = np.add(total_weights[i], weight_vector_for_ten_perceptron[i])

    for i in range(0, 10):
        total_weights[i][:] = [x/5 for x in total_weights[i]]
    print("     Accuracy for this pair is ", total_correct_percent / 5)

                        
                        
    #Test for each different parameter settting
    correct_num_on_test = 0
    number_of_instance_in_test_data = len(test_data)
    for test_instance in test_data:
        prediction_ten_perceptron = []
        for i in range(10):
            current_result = np.dot(total_weights[i], test_instance[0] + [1])
            prediction_ten_perceptron.append(current_result)
        maybe_correct_perceptron = np.argmax(prediction_ten_perceptron)
        if maybe_correct_perceptron == test_instance[1]:
            correct_num_on_test = correct_num_on_test + 1
    correct_percentage_on_test = float(correct_num_on_test / number_of_instance_in_test_data)
    print ("For:")
    print ("     current learning rate: ", each_parameter_comb[0])
    print ("     if decay: ", each_parameter_comb[1])
    print ("     how training set order: ", each_parameter_comb[2])
    print ("     how to initial weight: ", each_parameter_comb[3])
    print ("     epoch number: ", each_parameter_comb[4])
    print ("     Accuracy is ", correct_percentage_on_test * 100, "%")
    print ("                 ")
    print ("                 ")

    total_vector.append(correct_percentage_on_test)

best_para_index = np.argmax(total_vector)
print("The best parameter combo is: ")
print("     Learning Rate: ", parameter_list[best_para_index][0])
print("     if decay: ", parameter_list[best_para_index][1])
print("     how training set order: ", parameter_list[best_para_index][2])
print("     how to initial weight: ", parameter_list[best_para_index][3])
print("     Accuracy is: ", total_vector[best_para_index] * 100, "%")
