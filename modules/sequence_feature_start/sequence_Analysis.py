import pandas as pd
import numpy as np
import openpyxl
import csv
import os


def main():
    csv_to_pandad_df()
    get_data_excel()


def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
    return np.array([ltrdict[x] for x in seq])

# Old function
"""
def get_data_excel():
    "Load the data from an Excel worksheet"
    book = openpyxl.load_workbook('data/datasetWithNone.xlsx', data_only=True)
    df = pd.DataFrame()
    sheet = book["Sheet1"]
    with open('data/sigma_data.csv', 'w') as csvfile:
        csvfile.write("name,{}sigma\n".format("base,"*81))
    for row in sheet:
        base_list = []
        id = row[0].value
        sequence = row[2].value
        for base in sequence:
            base_list.append(base)
        sigma = row[1].value
        if sigma is None:
            sigma = "Not present"
        elif "," in sigma:
            print("more sigma:",sigma)
            sigma_list = sigma.split(",")
            for sigma in sigma_list:
                base_list = []
                id = row[0].value
                sequence = row[2].value
                for base in sequence:
                    base_list.append(base)
                print(sigma)
                base_list.append(str(sigma.strip("\n").strip(" ")))
                base_list.insert(0, str(id)+str(sigma[6:]))
                write_to_excel(base_list)
        else:        
            base_list.append(str(sigma.strip("\n")))
            base_list.insert(0, str(id))
            write_to_excel(base_list)

    book.close()
"""
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
        # machine_learn(df)
    except:
        print()


def machine_learn(df):
    y = df.iloc[:,-1]
    x = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    model = RandomForestClassifier(n_jobs=-1, n_estimators=1800, max_features=0.4, max_depth=46, max_leaf_nodes=40,
                           min_samples_leaf=0.05, min_samples_split=0.2)


if __name__ == "__main__":
    main()
