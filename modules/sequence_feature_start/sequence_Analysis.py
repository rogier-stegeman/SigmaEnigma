import pandas as pd
import numpy as np
import openpyxl
import csv

def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
    return np.array([ltrdict[x] for x in seq])

def get_data_excel():
    book = openpyxl.load_workbook('datasetMetNone.xlsx', data_only=True)
    df = pd.DataFrame()
    sheet = book["Sheet1"]
    for row in sheet:
        base_list = []
        id = row[0].value
        sequence = row[2].value
        for base in sequence:
            base_list.append(base)
        sigma = row[1].value
        if sigma is None:
            sigma = "Not present"
        base_list.append(str(sigma))
        base_list.insert(0, str(id))
        write_to_excel(base_list)

    book.close()


def write_to_excel(base_list):
    with open('sigma_data.csv', 'a+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow( base_list)
        csvfile.close()
        #csvfile.read(25)

def csv_to_pandad_df():
    try:
        df = pd.read_csv(r'C:\Users\damia\PycharmProjects\promotor_machine_learning\venv\sigma_data.csv')
        machine_learn(df)
    except:
        print()


def machine_learn(df):
    y = df.iloc[:,-1]
    x = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    model = RandomForestClassifier(n_jobs=-1, n_estimators=1800, max_features=0.4, max_depth=46, max_leaf_nodes=40,
                           min_samples_leaf=0.05, min_samples_split=0.2)
csv_to_pandad_df()






get_data_excel()
