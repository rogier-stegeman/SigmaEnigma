import pandas as pd
import numpy as np
import openpyxl
import csv
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing


def main():
    #get_data_excel()
    pandas_df = csv_to_pandad_df()
    machine_learn(pandas_df)



def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
    return np.array([ltrdict[x] for x in seq])


def get_data_excel():
    """Load the data from an Excel worksheet"""
    book = openpyxl.load_workbook('data/datasetWithNone.xlsx', data_only=True)
    df = pd.DataFrame()
    sheet = book["Sheet1"]
    with open('data/sigma_data.csv', 'w') as csvfile:
        csvfile.write("name,{}sigma\n".format("base,"*81))
    for row in sheet:
        base_list = []
        sigma_list = []
        sigma = row[1].value
        if sigma is None:
            sigma_list = ["Not present"]
        else:
            sigma_list = sigma.split(",")
        id = row[0].value
        sequence = row[2].value.lower()
        for base in sequence:
            base_list.append(base)
        for sigma in sigma_list:
            sigma = sigma.strip(" ")
            out_list = []
            if sigma == "none":
                out_list.append(str(id)+"s:no")
            else:
                out_list.append(str(id)+"s"+str(sigma[5:]))
            out_list.extend(base_list)
            out_list.append(str(sigma.strip("\n")))
            write_to_excel(out_list)

    book.close()

def write_to_excel(base_list):
    """Write processed data to csv file for import
    into Excel"""
    with open('data/sigma_data.csv', 'a+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        filewriter.writerow(base_list)
        #csvfile.read(25)


def csv_to_pandad_df():
    try:
        df = pd.read_csv('data/sigma_data.csv')
        return df
    except:
        print()


def machine_learn(df):
    seed = 7
    y = df.iloc[:,-1]
    cols = [col for col in df.columns if col not in ['name', 'sigma']]
    X = df[cols]
    model = RandomForestClassifier(n_jobs=-1, n_estimators=1800, max_features=0.4, max_depth=46, max_leaf_nodes=40,
                           min_samples_leaf=0.05, min_samples_split=0.2)
    #print(X)
    #print(y)
    #df.set_index('name')
    label_encoder = LabelEncoder()
    integer_encoded_label = label_encoder.fit_transform(y)
    integer_encoded_label = integer_encoded_label.reshape(len(integer_encoded_label), 1)
    #print(integer_encoded_label)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded_feature = onehot_encoder.fit_transform(X)

    # one hot the sequence

    # reshape because that's what OneHotEncoder likes
    #integer_encoded_feature = integer_encoded_feature.reshape(len(integer_encoded_feature), 1)
    #onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_feature)
    #print(  integer_encoded_feature)
    X = integer_encoded_feature
    y = integer_encoded_label.ravel()
    print(X)


    model.fit(X,y)

    kfold = model_selection.KFold(n_splits=4, random_state=seed)
    scoring = {'acc': 'accuracy',
               'f1': 'f1',
               'recall': 'recall',
               'avg_prec': 'average_precision',
               }

    scores = cross_validate(model, X, y, cv=kfold, scoring=scoring,
                            return_train_score=False)
    y_pred = model_selection.cross_val_predict(model, X, y, cv=kfold)
    for k, v in scores.items():
        print(k, v)
    y2 = y.values
    confusion = confusion_matrix(y2, y_pred)
    name = "randomforestclassifier"
    visualize_confusion_matrix(confusion, name)

def visualize_confusion_matrix(self, cm, name):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative', 'Positive']
    plt.title('Hold or sell classificatie')
    plt.ylabel(name)
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    



if __name__ == "__main__":
    main()
