import pickle
import pandas as pd
import numpy as np

from pprint import pprint

def main():
    with open('../../data/column_names_pickle_dummps.txt', "r") as f:
        columns = pickle.loads(f.read())
    columns.remove("ID")
    columns.remove("TARGET")
    var_dict = extract_variable(columns)
    for each_var in sorted(var_dict.keys()):
        print var_dict[each_var]


def extract_variable(columns):
    var_dict = {}
    for column in columns:
        items = column.split("_")
        for item in items:
            if "var" in item:
                if item in var_dict.keys():
                    var_dict[item].append(column)
                else:
                    var_dict[item] = [column]
    return var_dict





if __name__ == '__main__':
    main()