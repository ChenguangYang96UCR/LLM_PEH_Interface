import os
import csv 
import pandas as pd
import torch
from torch_geometric.data import Data
from retriever.lm_modeling import load_model, load_text2embedding
from retriever.retrieval import retrieval_via_pcst
import numpy as np
from sklearn.model_selection import train_test_split

path = 'retriever_graph'
model_name = 'sbert'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'
cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

def find_files(directory, filetype='docx'):
    docx_files = []
    sub_paths = []
    for filename in os.listdir(directory):

        if filename.endswith(f".{filetype}"):
            docx_files.append(filename)
    
    return docx_files


def extract_service_triples(text_file : str, service_type : str):

    """
    Extract service triples based on service type

    Args:
        text_file (str): file path
        service_type (str): food, shelter, mental_health
    """    

    os.makedirs('./graph/', exist_ok=True)
    os.makedirs('./graph/' + service_type + '/', exist_ok=True)

    work_path = os.path.abspath('.') + '/neo4j_store/'
    # data =[["subject", "relation", "object"]]
    data_subject = {}
    data_relation = {}
    data_object = {}
    services_list = []

    if service_type  == "food" :
        service_csv = pd.read_csv('./files/Final_Philadelphia_Emergency_Food_2025_0325.csv')
    
    if service_type == "shelter":
        service_csv = pd.read_csv('./files/Final_FindHelp_extracted_data_philadelphia_temporary_shelter_2025_0402.csv')

    if service_type == "mental_health":
        service_csv = pd.read_csv('./files/Final_FindHelp_extracted_data_philadelphia_mental_health_2025_0402.csv')

    for index, row in service_csv.iterrows():
        services_list.append(row['Service_Name'])
    
    if not os.path.exists(work_path + text_file):
        print(work_path + text_file)
        ValueError('Triples file {0} does not exist'.format(work_path + text_file))
    else:
        with open(work_path + text_file, 'r+', encoding='utf-8') as file:
            line = file.readline()
            while line:
                clear_line = line.strip('\n')
                # judge string is start with '[' and end with ']'
                if clear_line.startswith('[') and clear_line.endswith(']'):
                    # strip start and end character 
                    clear_line = clear_line.lstrip('[').rstrip(']')
                    triple = clear_line.split(';')

                    # triples len is longer than 3, then continue
                    if not len(triple) == 3:
                        line = file.readline()
                        continue
                    # get the triple's sunject
                    triple_subject = triple[0]
                    # get the triple's relation
                    triple_relation = triple[1].lstrip()
                    # get the triple's object
                    triple_object = triple[2].lstrip()
                    if triple_subject in services_list:
                        if not triple_subject in data_subject:
                            data_subject[triple_subject] = []
                            data_relation[triple_subject] = []
                            data_object[triple_subject] = []
                            data_subject[triple_subject].append(triple_subject)
                            data_relation[triple_subject].append(triple_relation)
                            data_object[triple_subject].append(triple_object)
                        else:
                            data_subject[triple_subject].append(triple_subject)
                            data_relation[triple_subject].append(triple_relation)
                            data_object[triple_subject].append(triple_object)
                    line = file.readline()
                else:
                    line = file.readline()
        data =[["subject", "relation", "object"]]
        for service in data_subject:
            subjects = data_subject[service]
            relations = data_relation[service]
            object = data_object[service]
            for index in range(len(subjects)):
                data.append([subjects[index],relations[index], object[index]])
        with open("./graph/" + service_type + "/graph.csv", mode="w", newline='', encoding='utf-8') as output:
            writer = csv.writer(output)
            writer.writerows(data)
        
def step_one():

    """
    Generate the node and edge information based on graph
    """

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)
    base_path = './graph/'
    service_type_list = ["food", "shelter", "mental_health"]
    for service_type in service_type_list:
        os.makedirs(path_nodes + '/' + service_type + '/', exist_ok=True)
        os.makedirs(path_edges + '/' + service_type + '/', exist_ok=True)
        work_path = base_path + service_type + '/'
        csv_files = find_files(work_path, filetype='csv')
        index = 0
        for csv_file in csv_files:
            with open(work_path + csv_file, mode='r+', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)  # Create a CSV reader object
                header = next(reader)  # Read the header row
                subject_index = header.index("subject")  # Get index of desired column
                relation_index = header.index("relation")
                object_index = header.index("object")
                nodes = {}
                edges = []
                for row in reader:
                    # judge string is start with '[' and end with ']'
                    subject = row[subject_index]
                    relation = row[relation_index]
                    object = row[object_index]
                    if subject not in nodes:
                        nodes[subject] = len(nodes)
                    if object not in nodes:
                        nodes[object] = len(nodes)
                    edges.append({'src': nodes[subject], 'edge_attr': relation, 'dst': nodes[object]})
                nodes = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
                edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

                nodes.to_csv(f'{path_nodes}/' + service_type + '/' + str(index) +'.csv', index=False)
                edges.to_csv(f'{path_edges}/' + service_type + '/' + str(index) +'.csv', index=False)
                index = index+1
            

def step_two():

    """
    Generate the nodes and edges' embedding 
    """

    print('Loading dataset...')
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    print('Encoding graphs...')
    os.makedirs(path_graphs, exist_ok=True)
    service_type_list = ["food", "shelter", "mental_health"]
    for service_type in service_type_list:
        print(f"Encoding {service_type} graphs")
        os.makedirs(path_graphs + '/' + service_type + '/', exist_ok=True)
        # nodes
        nodes = pd.read_csv(f'{path_nodes}/' + service_type + '/'+ f'{0}.csv')
        edges = pd.read_csv(f'{path_edges}/' + service_type + '/'+ f'{0}.csv')
        # nodes.node_attr.fillna("", inplace=True)
        nodes['node_attr'] = nodes['node_attr'].fillna("")
        if len(nodes) == 0:
            print(f'Empty {service_type} graph at 0')
            continue
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        # edges
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        torch.save(pyg_graph, f'{path_graphs}/' + service_type + '/' + f'{0}.pt')


#! In this preprocess, we consider all graph as one whole large graph
if __name__ == '__main__':
    work_path = os.path.abspath('./neo4j_store/') 
    txt_files = find_files(work_path, filetype='txt')
    # Seperate to different service graph based on service type
    for txt_file in txt_files:
        print('Write {0} file triples into csv file'.format(txt_file))
        extract_service_triples(txt_file, 'food')
        extract_service_triples(txt_file, 'shelter')
        extract_service_triples(txt_file, 'mental_health')

    #Embedding nodes and edges
    step_one()
    step_two()
