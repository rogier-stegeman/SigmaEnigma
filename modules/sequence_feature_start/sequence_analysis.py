import pandas as pd
import numpy as np
import openpyxl
import csv
import os
import boxFinder as bF
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
from sklearn.model_selection import learning_curve


def main():
    get_data_excel()
    pandas_df = csv_to_pandad_df()
    X,y = pre_process(pandas_df)
    #print(X)
   # print(y)
    models = create_models()
    test_multiple_models(models, X, y)

   # machine_learn(pandas_df)


def csv_to_pandas_df():
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
        model = [
            ('RF',
             RandomForestClassifier(n_jobs=-1, n_estimators=1800, max_features=0.4, max_depth=46, max_leaf_nodes=40,
                                    min_samples_leaf=0.05, min_samples_split=0.2))
        ]

        return model

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
        print(name)
        if name == "RF":
            model.fit(X, y)
            curve = plot_learning_curve(model,name,X,y)
            curve.show()
            show_tree(model, X, y)
            feature_selection(model, X)
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
    for col in X.columns:
        if not col.startswith("box"):
            print(col)
            X[col] = X[col].map(base_dict)

    label_encoder = LabelEncoder()
    integer_encoded_label = label_encoder.fit_transform(pre_y.values.ravel())
    integer_encoded_label = integer_encoded_label.reshape(len(integer_encoded_label), 1)
    y = integer_encoded_label.ravel()



    return X,y


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
    print(model)
    estimator = model.estimators_[5]


    print(list(X.columns))
#    print(list(y.columns))
    export_graphviz(estimator, out_file='tree.dot',
                feature_names=list(X.columns),
                class_names=['none','Sigma70'],
                rounded=True, proportion=False,
                precision=2, filled=True)

    os.environ['PATH'] = os.getcwd()+"\\tree.dot"
    print("hier")
    print(os.environ['PATH'])

# Convert to png using system command (requires Graphviz)

    print(os.getcwd())
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt
    

if __name__ == "__main__":
    main()