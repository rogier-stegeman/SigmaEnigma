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
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import re


def main():
    get_data_excel()
    pandas_df = csv_to_pandad_df()
    X,y = pre_process(pandas_df)
    #print(X)
   # print(y)
    models = create_models()
    test_multiple_models(models, X, y)

   # machine_learn(pandas_df)



def get_data_excel():
    """Load the data from an Excel worksheet"""
    book = openpyxl.load_workbook('data/datasetWithNoneSigma70.xlsx', data_only=True)
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
        #print(df)
        return df
    except:
        print()

def create_models():
        models = [
            ('LOR',
             LogisticRegression(n_jobs=-1, penalty='l1', solver='saga', multi_class='ovr', class_weight='balanced',
                                max_iter=4800)),
            ('KNN', KNeighborsClassifier(n_jobs=-1, n_neighbors=16, weights='distance', algorithm='auto')),
            ('RF',
             RandomForestClassifier(n_jobs=-1, n_estimators=1800, max_features=0.4, max_depth=46, max_leaf_nodes=40,
                                    min_samples_leaf=0.05, min_samples_split=0.2))
        ]
        return models

def test_multiple_models(models, X, y):
    # test multiple models. Optional function
    results = []
    names = []
    seed = 7
    mean = []
    std = []
    max = []
    min = []
    f1 = []
    recall = []
    precicion = []
    cm = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=4, random_state=seed)
        accuracy_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
        F1_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring="f1")
        recall_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring="recall")
        precicion_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring="average_precision")
        y_pred = model_selection.cross_val_predict(model, X, y, cv=kfold)

        y2 = y
        confusion = confusion_matrix(y2, y_pred)
        visualize_confusion_matrix(confusion,name)
        cm.append(confusion)

        results.append(accuracy_results)
        names.append(name)
        f1.append(F1_results.mean())
        recall.append(recall_results.mean())
        precicion.append(precicion_results.mean())
        mean.append(accuracy_results.mean())
        std.append(accuracy_results.std())
        max.append(accuracy_results.max())
        min.append(accuracy_results.min())

    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def pre_process(df):
    colsy = [col for col in df.columns if col in ['sigma']]
    print(colsy)
    colsx = [col for col in df.columns if col not in ['name', 'sigma']]
    print("he")
    X = df[colsx]
    pre_y = df[colsy]
    print(X)
    print(pre_y)
    base_dict = {
        "a": 0,
        "c": 1,
        "g": 2,
        "t": 3
    }
    for col in  X.columns :
        print(col)
        X[col] = X[col].map(base_dict)

    label_encoder = LabelEncoder()
    integer_encoded_label = label_encoder.fit_transform(pre_y.values.ravel())
    integer_encoded_label = integer_encoded_label.reshape(len(integer_encoded_label), 1)
    y = integer_encoded_label.ravel()



    return X,y

def boxFinder(seq):
    box10 = findRe(pattern='[TG]A[ACGT]{3}[AT]',seq=seq.upper()[35:60],start=True, add = 35)
    box35 = findRe(pattern='T[GT]{2}[ACGT]{3}',seq=seq.upper()[:35],start=False)
    box10Ext = False
    afstand = 0
    if not isinstance(box10,bool):
        box10Ext = findRe(pattern='G[ACTG][TG]A[ACGT]{3}[AT]',seq=seq.upper()[35:60],start=True, add = 35)
        if not isinstance(box35,bool):
            if not isinstance(box10Ext,bool):
                afstand = box10Ext - box35
            else:
                afstand = box10 - box35

    return ([not isinstance(box35,bool),
             not isinstance(box10Ext,bool),
             not isinstance(box10,bool),
afstand])


def findRe(pattern, seq, start, add = 0):
    try:
        x = re.search(pattern,seq)
        if start:
            return(x.start()+add)
        else:
            return(x.end()+add)
    except AttributeError as a:
        return(False)

def machine_learn(df):
    seed = 7
    colsy = [col for col in df.columns if col  in ['sigma']]
    colsx = [col for col in df.columns if col not in ['name', 'sigma']]

    X = df[colsx]
    #print( pre_X)
    pre_y = df[colsy]
    model = RandomForestClassifier(n_jobs=-1, n_estimators=1800, max_features=0.4, max_depth=46, max_leaf_nodes=40,
                           min_samples_leaf=0.05, min_samples_split=0.2)
    #df.set_index('name')
    base_dict= {
        "a": 0,
        "c": 1,
        "g": 2,
        "t": 3
    }
    for col in  X.columns :
        print(col)
        X[col] = X[col].map(base_dict)



    label_encoder = LabelEncoder()
    integer_encoded_label = label_encoder.fit_transform( pre_y .values.ravel())
    integer_encoded_label = integer_encoded_label.reshape(len(integer_encoded_label), 1)
    y = integer_encoded_label.ravel()
    #print(integer_encoded_label)
    '''

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded_feature = onehot_encoder.fit_transform(pre_X)
    '''
    
    # one hot the sequence

    # reshape because that's what OneHotEncoder likes
    #integer_encoded_feature = integer_encoded_feature.reshape(len(integer_encoded_feature), 1)
    #onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_feature)
    #print(  integer_encoded_feature)
    #X = integer_encoded_feature
    #print(X)

    #print(X[0,1:])
    #X =  pd.DataFrame(data=X)

    #print(y)

   # X = shuffle(X)
    #y = shuffle (y)
    #print(y)



    model.fit(X,y)
    #print(X)
    #print(pd.DataFrame(X))
    #print(X.shape)
    #print(y.shape)
    #print(type(X))
    #print(type(y))
    #print(X)
    #print(y)

    kfold = model_selection.KFold(n_splits=4, random_state=seed)
    scoring = {'accuracy':'accuracy',
               'f1_micro': 'f1_micro',
               'recall_micro': 'recall_micro',
               }

    scores = cross_validate(model, X, y, cv=kfold, scoring=scoring,
                            return_train_score=False)
    y_pred = model_selection.cross_val_predict(model, X, y, cv=kfold)
    for k, v in scores.items():
        print(k, v)
    y2 = y
    confusion = confusion_matrix(y2, y_pred)
    #print(list(X))
    #print(y)

    name = "randomforestclassifier"
    #feature_selection(model,X)
    show_tree(model,X,pre_y)
    #visualize_confusion_matrix(confusion, name)

def feature_selection(model,X):
    print(type(X))

    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index=X.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)
    importances = model.feature_importances_
    print(model.classes_)
    std = np.std([model.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()



def visualize_confusion_matrix( cm, name):
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

def show_tree(model,X,y):
    #Export as dot file
    estimator = model.estimators_[5]
    print(list(X.columns))
    print(list(y.columns))
    export_graphviz(estimator, out_file='tree.dot',
                feature_names=list(X.columns),
                class_names=['none','Sigma70'],
                rounded=True, proportion=False,
                precision=2, filled=True)

# Convert to png using system command (requires Graphviz)

    print(os.getcwd())
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    

if __name__ == "__main__":
    main()
