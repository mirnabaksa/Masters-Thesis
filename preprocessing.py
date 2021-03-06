from SignalDataset import StatsTestDataset

import numpy as np
import h5py
import csv
import pandas as pd
from os import listdir
from os.path import isfile, join

from statistics import mean, median, stdev
from scipy.stats import mode, skew, hmean, gmean, entropy, iqr

import sys
csv.field_size_limit(sys.maxsize)

def constructDatasetCSV(root_dir, dataset_name):
    print("Constructing csv...")
    labels = []
    with open('csv/' + dataset_name, 'w') as dataset_file:
        file_writer = csv.writer(dataset_file)
        file_writer.writerow(("file", "label"))
        for sub_dir in listdir(root_dir):
            label = sub_dir
            labels.append(label)
            #if (not ( label == "ecoli" or label == "pseudomonas_koreensis" or label == "pantonea_agglomerans" or label == "bacillus_anthracis")):
            #    continue
            
            target_dir = join(root_dir, sub_dir)
            for filename in listdir(target_dir):
                file_writer.writerow((join(target_dir, filename), label))

           
    df = pd.read_csv('csv/' + dataset_name)
    ds = df.sample(frac = 1)
    ds.to_csv('csv/' + dataset_name)
    print('csv/' + dataset_name + " created")
           
def constructRawSignalValuesCSV(dataset_csv,  name = 'raw_dataset.csv'):
    df_semantic = pd.read_csv('csv/' + dataset_csv) 
    column_names = ["label","raw_data"]

    read_len = 100000

    df = pd.DataFrame(columns=column_names)
    for i in range(len(df_semantic)):
        index, filename, label = df_semantic.loc[i]
        f = h5py.File(filename, 'r')
        data = np.array(f["Raw"]["Reads"]["Read_981"]["Signal"]).astype(np.float)

        for i in range(len(data)//read_len):
            data_chunk = data[i*read_len : (i+1)*read_len]
            data_str =  ','.join([str(num) for num in data_chunk])
            df = df.append({'raw_data' : data_str, 'label' : label}, ignore_index = True)
    
    df.to_csv('csv/' + name)
    print('csv/' + name + " created!")
    

from sklearn.preprocessing import minmax_scale, normalize

def constructStatsDataset(source = "raw_dataset.csv", dest = "stats_dataset.csv", chunk = 400):
    df_raw = pd.read_csv('csv/' + source)  
    column_names = ["label","stats_data"]
    df = pd.DataFrame(columns=column_names)
    labels = []
    for i in range(len(df_raw)):
        row = df_raw.loc[i]
        index, label, data = row
        if label not in labels:
            labels.append(label)

        s = 0
        splitted = data.split(",")
        splitted = [float(num) for num in splitted]
        
        stats_data = []
        for i in range(len(splitted)//chunk):
            data_chunk = splitted[i*chunk : (i+1)*chunk]
            if len(data_chunk) < 2:
                continue
            
            data_mean, data_gmean = mean(data_chunk), gmean(data_chunk)
            data_median = median(data_chunk)
            data_stdev = stdev(data_chunk)
            data_entropy = entropy(data_chunk)
            data_iqr = iqr(data_chunk)
            data_max, data_min = max(data_chunk), min(data_chunk)

            stats = [data_mean, data_gmean, data_median, data_stdev, data_entropy, data_iqr]
            stats_data.append(",".join([str(param) for param in stats]))
        
        df = df.append({'label' : label, 'stats_data' : '$'.join(stats_data)}, ignore_index = True)
    

    new_df = pd.DataFrame(columns = ["label","stats_data"])
    lens = []
    for label in labels:
        data = df.loc[df['label'] == label]
        lens.append(len(data))


    new_df = new_df.sample(frac=1).reset_index(drop=True)
    new_df.to_csv('csv/' + dest)
    print("csv/" + dest + " created!")

if __name__ == '__main__':
    #constructDatasetCSV("../Signals/perfect/", dataset_name = "dataset-perfect4.csv")
    #constructRawSignalValuesCSV('dataset-perfect4.csv', 'perfect-raw4.csv')
    #constructStatsDataset(source = 'perfect-raw4.csv', dest = 'perfect-stats-6class-test.csv')