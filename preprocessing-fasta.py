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
from SignalDataset import StatsDataset, TripletStatsDataset, StatsSubsetDataset
import torch
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

        if not (label == "ecoli" or label == "pseudomonas_koreensis" or label == "pantonea_agglomerans" or label == "yersinia_pestis"):
            continue

        if label not in labels:
            labels.append(label)

        splitted = data.split(",")
        splitted = [float(num) for num in splitted]
        
        stats_data = []
        if len(splitted)//chunk != 209:
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

shorter = ["Liste", "Enter", "Staph", "Salmo", "Esche", "Lacto", "Bacil"]
8875, 6454, 6295, 1996, 2095, 4065, 10406

from collections import Counter
labels = ["Enter", "Staph"]
def constructLomanStatsDataset(root_dir = "./loman/files/final/final-signals-short-filtered-4class.txt",  chunk = 400):
    num_samples = 4000
    c = Counter({'Enter': 0, 'Staph': 0, 'Liste': 0, 'Lacto' : 0, 'Bacil' : 0, 'Esche' : 0}) 

    df_raw = pd.read_csv(root_dir, sep=" ", header = None)  
    df_raw.columns = ["read_id", "num", "ch", "class", "raw"]
    column_names = ["label","stats_data"]

    df = pd.DataFrame(columns=column_names)
    print("data loaded")
    for j in range(len(df_raw)):
        row = df_raw.loc[j]
        _, _, _, label, raw_data = row
        
        if c["Enter"] == num_samples and c["Staph"] == num_samples and c["Liste"] == num_samples and c["Lacto"] == num_samples: # and c["Bacil"] == num_samples and c["Esche"] == num_samples:
            print(c)
            break

        if c[label] == num_samples:
            continue

        
        c[label] += 1
        
        splitted = raw_data.split(",")
        splitted = [float(num) for num in splitted]

        stats_data = []

        for i in range(len(splitted)//chunk):
            data_chunk = splitted[i*chunk : (i+1)*chunk]
            if len(data_chunk) < 2:
                continue
            
            data_mean = mean(data_chunk)
            data_median = median(data_chunk)
            data_stdev = stdev(data_chunk)
            data_iqr = iqr(data_chunk)
            stats = [data_mean, data_median, data_stdev, data_iqr]
            stats_data.append(",".join([str(param) for param in stats]))
        
        df = df.append({'label' : label, 'stats_data' : '$'.join(stats_data)}, ignore_index = True)

        if j % 1000 == 0:
            print(j)

    
    df.to_csv('csv/loman/4-class-4000.csv')
    print("csv/ created!")



count = 0
    
read_id = None
read_number = None
value = None
channel = None
df = pd.DataFrame(columns=['read_id', 'read_num', 'channel', 'value'])
def print_item(name, item):
        global count
        global df
        global read_id
        global read_number
        global value
        global channel
        
        if item.attrs:
            for key in item.attrs:
                if key == "read_id":
                    read_id = item.attrs[key]
                    read_id = read_id.decode('utf-8')
                    
                if key == "read_number":
                    read_number = item.attrs[key]

                if key == "channel_number":
                    channel = item.attrs[key]
                    channel = channel.decode('utf-8')

                    #print(read_id, read_number, channel, value[:10])
                    
                    df = df.append( {"read_id" : read_id, 'read_num' : read_number, 'value' : value, 'channel' : channel}, ignore_index = True)
                    
                    count += 1
                    value = None
                    read_id = None
                    read_number = None
                    channel = None

                    if count % 10000 == 0:
                        print(count)
                        df.to_csv("signals.csv", sep="\t", index = False, header = None)

                    if count == 50000:
                        print(count)
                        #df.to_csv("signals.csv", sep="\t", index = False, header = None)
                        exit(0)

       
        if hasattr(item, 'value'):
            value = item.value
            value = ','.join([str(num) for num in value])      
            
        

        
            
    

def log_hdf_file(hdf_file):
    print_item(hdf_file.filename, hdf_file)
    hdf_file.visititems(print_item)
    print(count)
    df.to_csv("signals.csv", sep="\t", index = False, header = None)


classes = ["listeria monocytogenes", "pseudomonas aeruginosa",  "enterococcus faecalis", "Staphylococcus aureus", "salmonella enterica", "escherichia coli" ]
#filter_func = lambda s : sum(s.startswith(x) for x in classes))
def filter_func(s):
    for x in classes:
        if s.startswith(x):
            return x
    return False

if __name__ == '__main__':
    constructLomanStatsDataset()
    exit(0)
    #with h5py.File('./loman/Zymo-GridION-EVEN-BB-SN-PCR-R10HC_multi/batch_0.fast5') as hdf_file:
    #    log_hdf_file(hdf_file)
    #exit(0)


    ''' df = pd.DataFrame(columns = ["read_id", "tax_id", "len", "class"]) 
    count = 0
    with open("./kraken/classified_reads.txt") as clas:
            for line in clas.readlines():
            classified, read_id, tax_id, leng, _ = line.split("\t")
            print(read_id)
            tax_id = tax_id.lower()
            read_class = filter_func(tax_id)
            if classified == "C" and read_class != False :
                df = df.append({'read_id' : read_id, 'tax_id' : tax_id, "len" : leng, "class" : read_class}, ignore_index = True)
            count += 1

            if(count % 100 == 0):
                print(count)
                df.to_csv("test.csv")
                exit(0)
        print(count)
        df.to_csv("classified_reads.csv")'''
    #print("Liste" in shorter)
    #exit(0)
    #df = pd.read_csv("./loman/files/final/final-signals-short.txt",  sep=" ", header = None)
    #df.columns = ["read_id", "read_num", "ch", "class", "value"]
    #df = df[df['class'].isin(shorter)]
    #print(len(df))
    #df.to_csv("./loman/files/final/final-signals-short-filtered.csv")
    df = pd.read_csv("./loman/files/final/final-signals-short.txt",  sep=" ", header = None)
    df.columns = ["read_id", "read_num", "ch", "class", "value"]
    #print(df)
    for short in shorter:
        print(len(df.loc[df['class'] == short]))

    #hf = h5py.File('data.h5', 'w')
    #hf.create_dataset('dataset', data=signal)
    #hf.close()
    
    #df = pd.read_csv('csv/parsed/stats_dataset-6class-400.csv') 
    #lab = df.loc[df['label'] == "klebsiella_"]
    #print(len(lab))
    #parser = ArgumentParser(add_help=False)
    #parser.add_argument('--filename', type=str)
    #parser.add_argument('--outfile', type=str)

    #params = parser.parse_args()
    #readFasta(params.filename, params.outfile)
    #constructDatasetCSV("../Signals/parsed_true/", dataset_name = "parsed/dataset-reads.csv")
    #constructRawSignaValuesCSV(root_dir = "parsed/dataset-reads.csv", dataset_name = 'parsed/raw_dataset.csv')
    #constructStatsDataset(root_dir = 'parsed/raw_dataset.csv', dataset_name = 'parsed/stats_dataset-6class-400.csv')
