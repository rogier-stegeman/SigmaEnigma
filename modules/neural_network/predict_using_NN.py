from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import pandas as pd
import numpy
from sklearn.preprocessing import OneHotEncoder


def base_to_int(base):
    d = {'a': '1', 'c': '2', 'g': '3', 't': '4'}
    return int(d[base])

def class_to_int(clss):
    d = {'Sigma70': '1', 'none': '2'}
    return int(d[clss])

def getdata():
    # load dataset
    dataset = pd.read_csv("data/sigma_data_validation.csv")
    dataset = dataset.fillna(0)
    # split into input (X) and output (Y) variables
    for col in dataset:
        if "base" in col:
            dataset[col] = dataset[col].apply(base_to_int)
    X = dataset.iloc[:,1:82]
    print(X)
    y = dataset.iloc[:,82] = dataset.iloc[:,82].apply(class_to_int)
    print(y)
    return X, y


def evalmodel(X, y, model):
    # evaluate the model
    correctly_identified = 0
    scores = model.evaluate(X, y)
    print(y.iloc[:10])
    print(model.predict(X.iloc[:10]).round(0).astype('int'))
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    correct = y.to_numpy()
    predicted = model.predict(X)
    #correct = correct[:5]
    predicted = predicted.round(0).astype('int')
    # predicted = predicted[:5].round(0).astype('int')
    # print(predicted2)
    for row_nr in range(len(correct)):
            # print(correct[row_nr][column_nr],">",predicted[row_nr][column_nr])
            if correct[row_nr] == predicted[row_nr][0]:
                correctly_identified += 1

    return correctly_identified/(len(correct))


def main():

    X,y = getdata()

    # load model
    model = load_model('model.h5')

    # evaluate the model
    correct = evalmodel(X,y,model)
    print("CORRECT%:",correct)

main()