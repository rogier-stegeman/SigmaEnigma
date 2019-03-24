# ©Damian Bolwerk , Alex Jansen , Rogier Stegemans
# Deze programma analyseerd sequenties en classifeerd deze met behulp van machine learning als sigma 70 bevattende sequentie of een sequentie
# die geen sigma 70 bevat.

# de nodige modules voor deze programma
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
pd.options.mode.chained_assignment = None

# De verschillende functies die nodig zijn om een bestand met sequenties te analyseren wordt hier aangeroepen.
def main():
    pandas_df = csv_to_pandas_df()
    X,y = pre_process(pandas_df)
    models = create_models()
    #machine_learn_only(models, X, y)
    test_multiple_models(models, X, y)

# leest een csv bestand in en zit het om in een pandas dataframe
def csv_to_pandas_df():
    try:
        df = pd.read_csv('data/sigma_data.csv')
        return df
    except:
        print()


# De verschilende machine learning algoritme die in de programma gebruikt worden
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

        return models
# traint de verschilende machine learning modelen en laat verschilende testen zien
# waaruit blijkt hoe goed elke algortime in staat is om sequenties met een siga 70
# accuraat te classificeren.
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
    precision = []
    cm = []
    for name, model in models:
        print(name)
        model.fit(X, y)
        if name == "RF":
            feature_selection(model, X)
            curve = plot_learning_curve(model, name, X, y)
            curve.show()
        kfold = model_selection.KFold(n_splits=4, random_state=seed)
        accuracy_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
        F1_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring="f1")
        recall_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring="recall")
        precision_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring="average_precision")
        y_pred = model_selection.cross_val_predict(model, X, y, cv=kfold)
        y2 = y
        confusion = confusion_matrix(y2, y_pred)
        visualize_confusion_matrix(confusion,name)
        cm.append(confusion)

        results.append(accuracy_results)
        names.append(name)
        f1.append(F1_results.mean())
        recall.append(recall_results.mean())
        precision.append(precision_results.mean())
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

# Deze functie traint alleen de modellen en laat minimale test resultaten zien en heeft daarom kortere runtime.
def machine_learn_only(model,x,y):
     for name, model in model:
         seed = 7
         model = model.fit(x, y)
         kfold = model_selection.KFold(n_splits=4, random_state=seed)
         scoring = {"accuracy": "accuracy",
                     "f1": "f1",
                     "recall":"recall",
                    "average_precision":"average_precision",
                     }
         scores = cross_validate(model, x, y, cv=kfold, scoring=scoring,
                                 return_train_score=False)
         for k, v in scores.items():
             print(k, v)

# Verwerkt de data van de pandas dataframe zodanig dat het gelezen kan worden door de machine
# learning algortime
def pre_process(df):
    colsy = [col for col in df.columns if col in ['sigma']]
    colsx = [col for col in df.columns if col not in ['name', 'sigma']]
    X = df[colsx]
    pre_y = df[colsy]
    base_dict = {
        "a": 0,
        "c": 1,
        "g": 2,
        "t": 3
    }
    for col in X.columns:
        if not col.startswith("box"):

            X[col] =X[col].map(base_dict)

    label_encoder = LabelEncoder()
    integer_encoded_label = label_encoder.fit_transform(pre_y.values.ravel())
    integer_encoded_label = integer_encoded_label.reshape(len(integer_encoded_label), 1)
    y = integer_encoded_label.ravel()
    X, y = shuffle(X,y, random_state=0)

    return X,y

# Laat de belang van elke feature zien waaruit de decision tress konden worden opgebouwd.

def feature_selection(model,X):
    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index=X.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)
    x_label = list( feature_importances.index)

    importances = model.feature_importances_
    std = np.std([model.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), x_label,rotation=90)
    plt.xlim([-1, X.shape[1]])
    
    plt.show()

# Laat zien hoe de verdeling is tussen de voorspellingen van de machine learning model
# en de daadwerkelijk klassen.

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

# Laat zien hoe goed de machine learning model presteerd op de trainings data en de validatie data naarmate
# de data waarop het getraind wordt groter wordt.
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