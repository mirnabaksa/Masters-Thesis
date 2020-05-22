from argparse import ArgumentParser

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import single_letter_alphabet

import csv
import h5py
import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

from sklearn.preprocessing import minmax_scale

ref_len = 83948
offset = 1000

def readFasta(filename, outfile):
    with open(filename, "rU") as handle:
        records = []
        for seq_record in SeqIO.parse(handle, "fasta"):
            id = seq_record.id

            print("############")
            print(seq_record.description)
            start_seq_len = len(seq_record)
            print(start_seq_len)
            i = 0
            for i in range(0, start_seq_len - ref_len + offset, offset):
                print("i" , i)
                chunk = seq_record[i: i+ref_len]
                print(len(chunk))
                rec = SeqRecord(Seq(str(chunk.seq), single_letter_alphabet), id=id + "|" + str(i), description=seq_record.description[:-18] + " offset " + str(i))
                print(rec)
                print("\n")
                
                records.append(rec)
                
                if len(chunk) < ref_len:
                    print("here")
                    break

            if start_seq_len < ref_len:
                records.append(seq_record)
                print("Len less")
                print(seq_record)
            print("\n")

        print("Num of records: ", len(records))
        SeqIO.write(records, outfile, "fasta")

def constructDatasetCSV(root_dir, dataset_name):
    print("Constructing csv...")
    labels = []
    with open('csv/' + dataset_name, 'w') as dataset_file:
        file_writer = csv.writer(dataset_file)
        file_writer.writerow(("file", "label"))
        for sub_dir in listdir(root_dir):
            label = sub_dir
            labels.append(label)

            target_dir = join(root_dir, sub_dir)
            for filename in listdir(target_dir):
                file_writer.writerow((join(target_dir, filename), label))

    print('csv/' + dataset_name + " created")

def constructRawSignalValuesCSV(root_dir, dataset_name):
    df_semantic = pd.read_csv('csv/' + root_dir) 
    column_names = ["label","raw_data"]

    df = pd.DataFrame(columns=column_names)
    for i in range(len(df_semantic)):
        filename, label = df_semantic.loc[i]

        f = h5py.File(filename, 'r')
        data = np.array(f["Raw"]["Reads"]["Read_981"]["Signal"]).astype(np.float)
        data_str =  ','.join([str(num) for num in data])
        df = df.append({'raw_data' : data_str, 'label' : label}, ignore_index = True)
    
    df.to_csv('csv/' + dataset_name)
    print('csv/' + dataset_name + " created!")


from statistics import mean, median, stdev
from scipy.stats import iqr

def constructStatsDataset(root_dir = "raw_dataset.csv", dataset_name = "stats_dataset.csv", chunk = 400):
    df_raw = pd.read_csv('csv/' + root_dir)  
    column_names = ["label","stats_data"]

    df = pd.DataFrame(columns=column_names)
    print("data loaded")
    labels = []
    for i in range(len(df_raw)):
        row = df_raw.loc[i]
        index, label, data = row

        #if not (label == "ecoli" or label == "pseudomonas_koreensis" or label == "pantonea_agglomerans" or label == "yersinia_pestis"):
        #    continue

        if label not in labels:
            labels.append(label)

        splitted = data.split(",")
        splitted = [float(num) for num in splitted]
        
        stats_data = []
        #print(len(splitted)//chunk)
        #print(ref_len//chunk)
        #print("Here")

        if len(splitted)//chunk != 209:
        #    print(len(splitted)//chunk)
        #    print(label)
        #    print()
            continue

        for i in range(len(splitted)//chunk):
            data_chunk = splitted[i*chunk : (i+1)*chunk]
            if len(data_chunk) < 2:
                continue
            
            data_mean = mean(data_chunk)
            data_median = median(data_chunk)
            data_stdev = stdev(data_chunk)
            data_iqr = iqr(data_chunk)
            #data_max, data_min = max(data_chunk), min(data_chunk)
            #data_mod = mode(data_chunk)[0][0]
            #data_skew = skew(data_chunk)
            stats = [data_mean, data_median, data_stdev, data_iqr]
            stats_data.append(",".join([str(param) for param in stats]))
        
        df = df.append({'label' : label, 'stats_data' : '$'.join(stats_data)}, ignore_index = True)
    

    new_df = df.sample(frac=1).reset_index(drop=True)
    new_df.to_csv('csv/' + dataset_name)
    print("csv/" + dataset_name + " created!")



def constructTripletDatasetFromEncodings():


if __name__ == '__main__':
    #parser = ArgumentParser(add_help=False)
    #parser.add_argument('--filename', type=str)
    #parser.add_argument('--outfile', type=str)

    #params = parser.parse_args()
    #readFasta(params.filename, params.outfile)
    #constructDatasetCSV("../Signals/parsed_true/", dataset_name = "parsed/dataset-reads.csv")
    #constructRawSignalValuesCSV(root_dir = "parsed/dataset-reads.csv", dataset_name = 'parsed/raw_dataset.csv')
    #constructStatsDataset(root_dir = 'parsed/raw_dataset.csv', dataset_name = 'parsed/stats_dataset-6class-400.csv')
