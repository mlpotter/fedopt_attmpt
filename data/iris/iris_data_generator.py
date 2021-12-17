from data.relational_table_preprocessor import relational_table_preprocess_dl
import pandas as pd
import numpy as np
import torch
from collections import Counter
import os

def load_iris(args):
    iris_df = pd.read_csv(os.path.join(args.datapath,"bezdekIris.data"), header=None)
    iris_df.loc[:,4] = iris_df.loc[:,4].map({"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2})
    label_list = iris_df.pop(4).tolist()
    data = iris_df.values

    args.class_num = 3
    print(iris_df.head())

    [_, _, _, _,_, train_data_local_dict, test_data_local_dict, args.class_num] = relational_table_preprocess_dl(args,
                                                                                                                 data,
                                                                                                                 label_list,
                                                                                                                 test_partition=0.2)
    for key in train_data_local_dict.keys():
        torch.save(train_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_train.pt"))
        print(dict(Counter(train_data_local_dict[key].dataset[:][1].numpy().tolist())))
        torch.save(test_data_local_dict[key], os.path.join(args.datapath,f"data_worker{key+1}_test.pt"))

if __name__ == '__main__':
    print("Nothing")