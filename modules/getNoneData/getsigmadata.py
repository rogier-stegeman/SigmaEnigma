import pandas as pd
import numpy as np
import openpyxl
import csv
import os
def main():
    get_data_excel()


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

main()