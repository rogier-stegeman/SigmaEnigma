from itertools import product
import json
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense
import random
import pandas as pd
import numpy
import numpy as np

# For conf matrix
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix


def base_to_int(base):
    """Convert dna bases to integers"""
    d = {'a': '1', 'c': '2', 'g': '3', 't': '4'}
    return int(d[base])


def class_to_int(clss):
    """Convert sigma classes to integers"""
    d = {'Sigma70': '1', 'none': '2'}
    return int(d[clss])


def getdata(datafile):
    """Load the dataset from a csv file."""
    # Read dataset
    dataset = pd.read_csv(datafile)
    # Use 0's where values are empty
    dataset = dataset.fillna(0)
    # Transforms all bases columns to integers
    for col in dataset:
        if "base" in col:
            dataset[col] = dataset[col].apply(base_to_int)
    # Split into input (X) and output (Y) variables
    X = dataset.iloc[:,1:86]
    y = dataset.iloc[:,86] = dataset.iloc[:,86].apply(class_to_int)
    return X, y


def create_model(X, y, layerset, loss, optimizer, epoch, seed):
    """Create, compile and fit a new model"""
    print("")
    # seed = 3782382748
    # Set the seed, instead of using a random one
    numpy.random.seed(seed)

    # Instantiate a new model
    model = Sequential()

    # Input layer
    model.add(Dense(85, input_dim=85, activation='relu'))

    # Hidden layers
    for layer in layerset:
        model.add(Dense(layer[0], activation=layer[1]))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    # Fit the model
    model.fit(X, y, epochs=epoch, batch_size=20)
    return model


def evalmodel(X, y, model, cm_choice):
    """Evaluate the model"""
    correctly_identified = 0
    correct = y.to_numpy()
    predicted = model.predict(X)
    predicted = predicted.round(0).astype('int')
    for row_nr in range(len(correct)):
            if correct[row_nr] == predicted[row_nr][0]:
                correctly_identified += 1
    
    if cm_choice.startswith("y"):
        # Get confusion Matrix
        y_pred = [l[0] for l in predicted]
        confusion = confusion_matrix(correct, y_pred)
        visualize_confusion_matrix(confusion, "NN")
    
    return correctly_identified/(len(correct))


def visualize_confusion_matrix(cm, name):
    """Use matplotlib to create a confusion matrix"""
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative', 'Positive']
    plt.title('Sigma')
    plt.ylabel(name)
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    plt.show()


def chunks(l, n):
    """Returns chunks"""
    for i in range(0, len(l), n):
        yield list(l[i:i+n])


def new_model(start_cycle=0):
    # Retrieve settings from a file
    with open('modules/neural_network/NNsettings.json', 'r') as f:
        config = json.load(f)
    presets = config.keys()
    choice = ""
    while choice not in presets: 
        choice = input(f"\nWhich settings would you like to use?\nChoose from: {presets}\n>>>")
    preset = config[choice]
    hidden_layer_nodes_list, layer_list, loss_list, optimizer_list, activation_list, epoch_list, seeds, write_score, stop_score, equal_layers, pre_layers, use_pre = preset["hidden_layer_nodes_list"], preset["layer_list"], preset["loss_list"], preset["optimizer_list"], preset["activation_list"], preset["epoch_list"], preset["seeds"], preset["write_score"], preset["stop_score"], preset["equal_layers"], preset["pre_layers"], preset["use_pre"]
    if pre_layers:
        total_tests = len(pre_layers)*len(loss_list)*len(optimizer_list)*len(epoch_list)*seeds
    else:    
        if equal_layers:
            total_tests = len(hidden_layer_nodes_list)*len(loss_list)*len(optimizer_list)*len(activation_list)*len(epoch_list)*len(layer_list)*seeds
        else:
            total_tests = sum([(len(activation_list)*len(hidden_layer_nodes_list))**layer_nr for layer_nr in layer_list])*len(optimizer_list)*len(epoch_list)*seeds
    print("\nSettings:")
    for k, v in preset.items():
        print(f"\t{k}: {v}")
    print("Total amount of tests:",total_tests)
    cm_choice = ""
    while not cm_choice.startswith(("y", "n")):
        cm_choice = input("\nShow confusion matrices? (Y/N)\n>>>").lower()
    print("")
    with open("results.csv", "w") as results:
        results.write("layerset,loss,optimizer,epoch,seed,layer,correct\n")
        X, y = getdata("data/sigma_data2.csv")

        # Defining an inner function to enable breaking all loops with a return statement, 
        # without having to pass all the variables to a new function.
        def run_all():
            count = 1
            layers = []
            if pre_layers:
                layers = pre_layers
            # Create all possible layers
            else:
                if equal_layers == True:
                    for hidden_layer_nodes in hidden_layer_nodes_list:
                        for layer in layer_list:
                            for activation in activation_list:
                                layers.append([[hidden_layer_nodes,activation]]*layer)
                else:
                    for layer in layer_list:
                        a = [hidden_layer_nodes_list,activation_list]
                        n_tuples = list(product(*a, repeat=layer))
                        for n_tuple in n_tuples:
                            layers.append(list(chunks(n_tuple,2)))
            print("Amount of hidden layer setups:",len(layers))
            for layerset in layers:
                for loss in loss_list:
                    for optimizer in optimizer_list:
                        for epoch in epoch_list:
                            for _ in range(seeds):
                                if count <= start_cycle:
                                    pass
                                else:
                                    seed = random.randint(1,(2**32) -1)
                                    seed = 3782382748
                                    model = create_model(X, y, layerset, loss, optimizer, epoch, seed)
                                    model.save("temp.h5")
                                    K.clear_session()
                                    # correct = evalmodel(X, y, model, cm_choice)
                                    # print("old:",correct)
                                    correct = validate_model("temp.h5", cm_choice)
                                    layerset_s = str(layerset).replace(",",";")
                                    options = f"{layerset_s},{loss},{optimizer},{epoch},{seed},{len(layerset)},{correct}"
                                    print(options)
                                    print(f"Test {count} of {total_tests} has an accuracy of {correct}")
                                    if correct >= write_score:
                                        results.write(f"{options}\n")
                                    if correct >= stop_score:
                                        print("The stop score was reached!")
                                        # return
                                count += 1
            print("The stop score was not reached (LIES)")

        run_all()    
        print("The last model was saved as temp.h5\nThe results have been written to results.csv")


def validate_model(model_name, cm_choice):
    X,y = getdata("data/sigma_data_test2.csv")
    # load model
    model = load_model(f'{model_name}') #BOTTLENECK
    # evaluate the model
    correct = evalmodel(X, y, model, cm_choice)
    K.clear_session()
    return correct


def main():
    # Startup menu
    choice = 0
    while choice not in ["1","2","3","4"]:
        choice = input("Enter your choice:\n1. Run training process\n2. Validate model\n3. Continue with previous training cycle\n4. Help\n>>>")
    
    if choice == "1":
        new_model(start_cycle=0)
    
    elif choice == "2":
        model_name = input("Enter the model file name (e.g. 'temp.h5'):\n>>>")
        cm_choice = ""
        while not cm_choice.startswith(("y", "n")):
            cm_choice = input("\nShow confusion matrix? (Y/N)\n>>>").lower()
        print("")
        correct = validate_model(model_name, cm_choice)
        print("CORRECT%:",correct)
    
    elif choice == "3":
        nr = int(input("Continue at cycle nr?\n>>>")) - 1
        new_model(start_cycle=nr)

    else:
        print("Sorry, this function is not available yet")


if __name__ == "__main__":
    main()