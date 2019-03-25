from itertools import product
import json
import keras
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
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
from statistics import mean

# Split train set into train and test
# Keras' accuracy doesn't make sense, calculate your own accuracy by comparing predicted with desired scores

def base_to_int(base):
    d = {'a': '1', 'c': '2', 'g': '3', 't': '4'}
    return int(d[base])

def class_to_int(clss):
    d = {'Sigma70': '1', 'none': '2'}
    return int(d[clss])

def getdata(datafile):
    # load dataset
    dataset = pd.read_csv(datafile)
    dataset = dataset.fillna(0)
    # split into input (X) and output (Y) variables
    for col in dataset:
        if "base" in col:
            dataset[col] = dataset[col].apply(base_to_int)
    X = dataset.iloc[:,1:86]
    # print(X)
    y = dataset.iloc[:,86] = dataset.iloc[:,86].apply(class_to_int)
    # print(y)
    return X, y


def create_model(X, y):
    """Create, compile and fit a new model"""
    # seed = random.randint(1,(2**32) -1)
    seed = 1406740558
    # Set the seed, instead of using a random one
    numpy.random.seed(seed)

    # Instantiate a new model
    model = Sequential()

    # Input layer
    model.add(Dense(85, input_dim=85, activation='relu'))
    
    # Hidden layers
    model.add(Dense(42, activation='relu', kernel_regularizer=keras.regularizers.l2(0.010)))
    model.add(Dropout(0.2))
    model.add(Dense(42, activation='linear', kernel_regularizer=keras.regularizers.l2(0.010)))
    model.add(Dropout(0.2))

    # Ouput layer
    model.add(Dense(1, activation='linear'))

    # Compile model
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
    model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
    # Fit the model
    model.fit(X, y, epochs=35, batch_size=20)
    return model, seed


def evalmodel(X, y, model, cm_choice):
    """Get the accuracy"""
    correctly_identified = 0
    scores = model.evaluate(X, y)
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



def new_model(cm_choice):
    """Create a new model"""
    with open("results_simple.csv", "a+") as results:
        results.write("seed,correct\n")
        X, y = getdata("data/sigma_data2.csv")
        model, seed = create_model(X, y)
        model.save("tempsimple.h5")
        K.clear_session()
        correct = validate_model("tempsimple.h5", cm_choice)
        print("Score:", correct)
        results.write(f"{seed}, {correct}")
        K.clear_session()
        return correct



def validate_model(model_name, cm_choice):
    """Evaluate the model"""
    X,y = getdata("data/sigma_data_test2.csv")
    # load model
    model = load_model(f'{model_name}')
    # evaluate the model
    correct = evalmodel(X, y, model, cm_choice)
    return correct


def main():
    # Startup menu
    choice = 0
    while choice not in ["1","2","3"]:
        choice = input("Enter your choice:\n1. Run training process\n2. Validate model\n3. Help\n>>>")

    if choice == "1":
        cm_choice = ""
        while not cm_choice.startswith(("y", "n")):
            cm_choice = input("\nShow confusion matrices? (Y/N)\n>>>").lower()
        print("")
        corrects = []
        try:
            os.remove("results_simple.csv")
        except:
            pass
        for i in range(5):
            corrects.append(new_model(cm_choice))
        print(f"Test has an accuracy of {corrects} = {mean(corrects)}")

    elif choice == "2":
        model_name = input("Enter the model file name (e.g. 'temp.h5'):\n>>>")
        cm_choice = ""
        while not cm_choice.startswith(("y", "n")):
            cm_choice = input("\nShow confusion matrix? (Y/N)\n>>>").lower()
        print("")
        correct = validate_model(model_name, cm_choice)
        print("CORRECT%:",correct)

    else:
        print("Sorry, this function is not available yet")
        


if __name__ == "__main__":
    main()