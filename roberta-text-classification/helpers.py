import os
from bs4 import BeautifulSoup
import csv
import matplotlib.pyplot as plt
import numpy as np 
import torch 
from transformers import RobertaTokenizer

def get_filenames(dataset_path):
    filenames = []
    files = os.walk(dataset_path)
    for f in files:
        for filename in f[2]:
            if filename[len(filename)-3:] == 'sgm':
                filenames.append(filename)
    return filenames

def build_dataset_dictionaries(dataset_path, filenames, all_topics):
    num_labels = len(all_topics)
    dataset_train_dict = {"text":[]}
    dataset_test_dict = {"text":[]}
    for topic in all_topics:
        dataset_train_dict[topic] = []
        dataset_test_dict[topic] = []
    for filename in filenames:
        print(filename)
        file = open(os.path.join(dataset_path, filename), 'rb')
        data = file.read()
        soup = BeautifulSoup(data)
        articles = [article for article in soup.findAll('reuters')]
        for article in articles:
            text = ''
            text = article.find('text').text
            brief = article.find('text', attrs={'type': 'BRIEF'})
            if brief:
                text = brief.title.text
            text = text.replace("\n", " ")
            if article['lewissplit'] == 'TRAIN':
                dataset_train_dict['text'].append(text)
            elif article['lewissplit'] == 'TEST':
                dataset_test_dict['text'].append(text)    
            topic_list = [topic.text for topic in article.topics.findAll('d')]
            for topic in all_topics:
                if topic in topic_list:
                    if article['lewissplit'] == 'TRAIN':
                        dataset_train_dict[topic].append(1)
                    elif article['lewissplit'] == 'TEST':
                        dataset_test_dict[topic].append(1)            
                else:
                    if article['lewissplit'] == 'TRAIN':
                        dataset_train_dict[topic].append(0)
                    elif article['lewissplit'] == 'TEST':
                        dataset_test_dict[topic].append(0)  
    dataset_val_dict = {}
    for key in dataset_train_dict.keys():
        dataset_val_dict[key] = dataset_train_dict[key][:2000]
        dataset_train_dict[key] = dataset_train_dict[key][2000:]
    
    return dataset_train_dict, dataset_val_dict, dataset_test_dict

def write_to_csv(dataset_path, all_topics, dataset_train_dict, dataset_eval_dict, dataset_test_dict):
    csv_columns = ['idx', 'text']
    for topic in all_topics:
        csv_columns.append(topic)
    csv_train = "reuters_train.csv"
    csv_eval = "reuters_eval.csv"
    csv_test = "reuters_test.csv"
    with open(os.path.join(dataset_path, csv_train), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_columns)
        for i in range(len(dataset_train_dict["text"])):
            row = [i, dataset_train_dict["text"][i]]
            for topic in all_topics:
                row.append(dataset_train_dict[topic][i])
            writer.writerow(row)

    with open(os.path.join(dataset_path, csv_eval), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_columns)
        for i in range(len(dataset_eval_dict["text"])):
            row = [i, dataset_eval_dict["text"][i]]
            for topic in all_topics:
                row.append(dataset_eval_dict[topic][i])
            writer.writerow(row)
            
    with open(os.path.join(dataset_path, csv_test), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_columns)
        for i in range(len(dataset_test_dict["text"])):
            row = [i, dataset_test_dict["text"][i]]
            for topic in all_topics:
                row.append(dataset_test_dict[topic][i])
            writer.writerow(row)
            
    return csv_train, csv_eval, csv_test

def show_histogram_multilabel(all_topics, raw_datasets):
    labels = all_topics
    histogram = {}
    for label in labels:
        histogram[label] = {0:0, 1:0}

    for key in raw_datasets.keys():
        columns = raw_datasets[key].column_names
        for row in raw_datasets[key]:
            for label in labels:
                label_value = raw_datasets[key][row['idx']][label]
                if(label_value == 0):
                    histogram[label][0] += 1
                else:
                    histogram[label][1] += 1
        histogram_proportion = {}
    for label in labels:
        histogram_proportion[label] = float(histogram[label][1] / histogram[label][0])
    plt.yticks(np.arange(0,1,0.01))
    plt.bar(histogram_proportion.keys(), histogram_proportion.values(), width = 0.5, color='g')
    
def show_histogram_binary(topic, raw_datasets):
    histogram = {"0":0, "1":0}
    for row in raw_datasets['train']:
        if(row[topic] == 0):
            histogram["0"] += 1
        else:
            histogram["1"] += 1
    plt.bar(histogram.keys(), histogram.values(), width=0.5, color='g')

def tokenize_function(sequences):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return tokenizer(sequences["text"], truncation=True)

def adjust_and_tokenize_datasets(raw_datasets): 
    for key in raw_datasets.keys():
        all_labels = []
        columns = raw_datasets[key].column_names
        for row in raw_datasets[key]:
            labels = [row[column] for column in columns if (column != 'idx' and column != 'text')]
            all_labels.append(labels)
        raw_datasets[key] = raw_datasets[key].add_column('labels', all_labels)
    columns = raw_datasets["train"].column_names
    columns.remove("labels")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=columns)
    tokenized_datasets.set_format("torch")
    tokenized_datasets = (tokenized_datasets
              .map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
              .rename_column("float_labels", "labels"))
    
    return tokenized_datasets