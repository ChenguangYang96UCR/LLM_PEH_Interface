import re
import streamlit as st
import openai
from py2neo import Graph
from deep_translator import GoogleTranslator
import logging
from retriever.lm_modeling import load_model, load_text2embedding
import os
import csv 
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from PIL import Image
from retriever.retrieval import retrieval_via_pcst

import huggingface_hub
import torch
import os
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from huggingface_hub import login


login(token = "")

# Define Order Method Global Value
GOOGLE_RATING = 1
DISTANCE_ORDER = 2

#* Define global value
google_translator_max_char = 5000
path = 'retriever_graph'
embeding_model_name = 'sbert'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'
cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

def switch_page():

    """
    switch search method
    """

    if st.session_state.page == "g_retriever":
        st.session_state.page = "cypher"
    else:
        st.session_state.page = "g_retriever"


def extract_weekday(input_string : str):

    """
    Extract weekday based on input

    Args:
        input_string (string): input string 

    Returns:
        weekday_match: extracted weekday 
    """    

    # extract weekday from string
    weekday_match = re.search(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", input_string)
    return weekday_match.group(1) if weekday_match else None

# * Get service open time from graph
def get_services_time(weekday, relation='xmlschema11-2#time'):

    """
    Get service open time from graph (cypher search)

    Args:
        weekday (str): weekday string
        relation (str, optional): relationship define. Defaults to 'xmlschema11-2#time'.

    Returns:
        triples: service triples
    """    

    triples = []
    graph = Graph(
            "bolt://localhost:7687", 
            auth=("neo4j", "123456789")
        )
    Query = 'MATCH (m:node)-[r]-(n:node) where type(r)=~\".*(?i){0}.*\" and n.name=~\".*(?i){1}.*\" RETURN m.name,type(r),n.name'.format(relation, weekday)
    query_result = graph.run(Query)
    for triple in query_result:
            triples.append(str(triple).replace('\t', ','))
    return triples

# * wild search from graph
def wild_search_by_keywords(key_word, relation=''):

    """
    Cypher wild search graph function 

    Args:
        key_word (string): key word for search
        relation (str, optional): triple's relation to search. Defaults to ''.

    Returns:
        list: search result for triple list
    """    

    triples = []
    if key_word == '':
        print('Error: search_node_by_keyword input \'key_word\' is nothing')
        return triples
    graph = Graph(
            "bolt://localhost:7687", 
            auth=("neo4j", "123456789")
        )
    if relation == '':
        # Query = 'MATCH (m:node)-[r]-(n:node) where m.name=~\".*(?i){0}.*\" or n.name=~\".*(?i){0}.*\" RETURN m.name,type(r),n.name'.format(key_word)
        Query = 'MATCH (m:node {{name:"{0}"}})-[r]-(n:node) RETURN m.name,type(r),n.name'.format(key_word)
        query_result = graph.run(Query)
        for triple in query_result:
            triples.append(str(triple).replace('\t', ','))
    else:
        Query = 'MATCH (m:node {{name:"{0}"}})-[r]-(n:node) RETURN m.name,type(r),n.name'.format(key_word)
        # Query = 'MATCH (m:node)-[r]-(n:node) where m.name=~\".*(?i){0}.*\" or n.name=~\".*(?i){0}.*\" RETURN m.name,type(r),n.name'.format(key_word, relation)
        query_result = graph.run(Query)
        for triple in query_result:
            triples.append(str(triple).replace('\t', ','))
    return triples

# * Get and combine services' information 
@st.cache_resource(show_spinner=False)
def getQuestion_answer(Service_list, loc, st,language = 'en'):

    """
    Get question's answer from LLM

    Args:
        Service_list (list): service name list [[service_name, google_rating]]
        st (class): streamlit class point
        language (str, optional): language used to show in the interface. Defaults to 'en'.
    """    

    all_triples = []    
    all_information = []
    for service_index,  Service in enumerate(Service_list):
        triples = wild_search_by_keywords(Service[0])
        slice_triples = [triples[i:i+100] for i in range(0, len(triples), 100)]
        all_response = []
        if not len(triples) == 0:
            all_triples.extend(triples)
            index = 1
            for slice_triple in slice_triples:
                if index > 2:
                    break
                answer_prompt = f"""
You are a social science expert, and you need to perform the following two steps step by step -  
Step1: Given the input knowledge graph triples (i.e., with the format (subject, relation, object)), 
please output corresponding natural language sentences about introduction and suggestion based on all knowledge graph triples.
input knowledge graph triples: {slice_triple}
"""
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # Updated to use the latest and more advanced model
                    messages=[
                        {"role": "user", "content": answer_prompt}
                    ],
                    temperature=0.2
                )
                all_response.append(response)
                index = index + 1 

        combine_prompt = f"""
Please combine these response, construct corresponding natural language sentences about Service Name, Address, Contract Method, Opening Hour, Brief Introduction and Transportation (my location is [{loc[0]},{loc[1]}], and service location is [{Service[2]}, {Service[3]}], please tell me the time required for walking and bus. And do not shgow the coordinate in the response). Show these information as a list and bold these titles. 
response: {all_response}
"""
        response = openai.ChatCompletion.create(
                        model="gpt-4o",  # Updated to use the latest and more advanced model
                        messages=[
                            {"role": "user", "content": combine_prompt}
                        ],
                        temperature=0.2
                    )
        st.markdown('''##### :blue[''' + str(f"({service_index + 1})") + ' ' + Service[0] + ''']''')
        if os.path.exists(f"street_images/{Service[0]}.png"):
            img = Image.open(f"street_images/{Service[0]}.png")
            st.image(img)
        # st.write(f"**{Service[0]}**")
        # google_rating = f"**Google Rating:{Service[1]}**"
        # google_rating_trans = GoogleTranslator(source='auto', target=language).translate(str(google_rating))
        # st.write(google_rating_trans, "\n")
        service_information = response.choices[0].message['content']
        if len(service_information) > google_translator_max_char:
            service_information = service_information[:google_translator_max_char]
        service_information_trans = GoogleTranslator(source='auto', target=language).translate(str(service_information))
        st.write(service_information_trans)

    # st.write(f"###{Service[0]} Google Rating:{Service[1]}", "\n", response.choices[0].message['content'])
    print("Finished GetQuestion_Answer!")
    return all_information

def extract_service_serving_name_from_triple(triple):

    """
    extract service name from triple

    Args:
        triple (string): triple string

    Returns:
        string: extracted service name
    """    

    # start_quotation = triple.find('"')
    target_index = triple.find('\'audience\'', 1)
    extract_service_name = triple[0:target_index - 1]
    if extract_service_name[0] == '\'':
        extract_service_name = extract_service_name.rstrip('\'')
        extract_service_name = extract_service_name.lstrip('\'')
    else:
        extract_service_name = extract_service_name.rstrip('\"')
        extract_service_name = extract_service_name.lstrip('\"')
    return extract_service_name   

def extract_type_service_name_from_triple(triple):

    """
    extract service name from triple

    Args:
        triple (string): triple string

    Returns:
        string: extracted service name
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

    # start_quotation = triple.find('"')
    target_index = triple.find('\'service type\'', 1)
    extract_service_name = triple[0:target_index - 1]
    if extract_service_name[0] == '\'':
        extract_service_name = extract_service_name.rstrip('\'')
        extract_service_name = extract_service_name.lstrip('\'')
    else:
        extract_service_name = extract_service_name.rstrip('\"')
        extract_service_name = extract_service_name.lstrip('\"')
    return extract_service_name   

def extract_time_service_name_from_triple(triple):
    target_index = triple.find('\'xmlschema11-2#time\'', 1)
    extract_service_name = triple[0:target_index - 1]
    if extract_service_name[0] == '\'':
        extract_service_name = extract_service_name.rstrip('\'')
        extract_service_name = extract_service_name.lstrip('\'')
    else:
        extract_service_name = extract_service_name.rstrip('\"')
        extract_service_name = extract_service_name.lstrip('\"')
    return extract_service_name   

def extract_open_time_from_triple(triple):
    clear_triple = str(triple).replace('\t', ',')
    target_index = clear_triple.find('xmlschema11-2#time', 1)
    extract_open_time = clear_triple[target_index + 19 :-1]
    hour_pattern = r"\b(\d{2}):\d{2}\b"

    # Extract hour part
    hours = [int(hour) for hour in re.findall(hour_pattern, extract_open_time)]
    return hours


def get_services_time(day_of_week, service_time, logger, relation='xmlschema11-2#time'):
    print('get_services_time')
    services_name = []
    graph = Graph(
            "bolt://localhost:7687", 
            auth=("neo4j", "123456789")
        )
    Query = 'MATCH (m:node)-[r]-(n:node) where type(r)=~\".*(?i){0}.*\" and n.name=~\".*(?i){1}.*\" RETURN m.name,type(r),n.name'.format(relation, day_of_week)
    query_result = graph.run(Query)
    for triple in query_result:
        raw_triple = str(triple).replace('\t', ',')
        open_time = extract_open_time_from_triple(triple)
        if len(open_time) >= 2:
            if service_time >= open_time[0] and service_time <= open_time[1]:
                extract_service_name =  extract_time_service_name_from_triple(raw_triple)
                services_name.append(extract_service_name)
        else:
            continue
    logger.debug(f"service list based on time search: {services_name}")
    return services_name

# * receive four kinds of serving type
def get_service_serving(*args, logger, relation='audience'):
    services_name = []
    graph = Graph(
            "bolt://localhost:7687", 
            auth=("neo4j", "123456789")
        )
    if len(args) == 1:
        Query = 'MATCH (m:node)-[r]->(n:node) where type(r)=~\".*(?i){0}.*\" and n.name=~\".*(?i){1}.*\" RETURN m.name,type(r),n.name'.format(relation, args[0])
    if len(args) == 2:
        Query = 'MATCH (m:node)-[r]->(n:node) where type(r)=~\".*(?i){0}.*\" and (n.name=~\".*(?i){1}.*\" or n.name=~\".*(?i){2}.*\") RETURN m.name,type(r),n.name'.format(relation, args[0], args[1])
    if len(args) == 3:
        Query = 'MATCH (m:node)-[r]->(n:node) where type(r)=~\".*(?i){0}.*\" and (n.name=~\".*(?i){1}.*\" or n.name=~\".*(?i){2}.*\" or n.name=~\".*(?i){3}.*\") RETURN m.name,type(r),n.name'.format(relation, args[0], args[1], args[2])
    if len(args) == 4:
        Query = 'MATCH (m:node)-[r]->(n:node) where type(r)=~\".*(?i){0}.*\" and (n.name=~\".*(?i){1}.*\" or n.name=~\".*(?i){2}.*\" or n.name=~\".*(?i){3}.*\" or n.name=~\".*(?i){4}.*\") RETURN m.name,type(r),n.name'.format(relation, args[0], args[1], args[2], args[3])

    query_result = graph.run(Query)
    for triple in query_result:
        raw_triple = str(triple).replace('\t', ',')
        extract_service_name =  extract_service_serving_name_from_triple(raw_triple)
        services_name.append(extract_service_name)
    unique_service = list(set(services_name))
    logger.debug(f"service list based on serving search: {unique_service}")
    return unique_service

def get_services_type(service_type, logger, relation='service type'):
    print('get_services_type')
    services_name = []
    graph = Graph(
            "bolt://localhost:7687", 
            auth=("neo4j", "123456789")
        )
    Query = 'MATCH (m:node)-[r]->(n:node) where type(r)=~\".*(?i){0}.*\" and n.name=~\".*(?i){1}.*\" RETURN m.name,type(r),n.name'.format(relation, service_type)
    # MATCH (m:node)-[r]-(n:node) where type(r)=~".*(?i)service type.*" and n.name=~".*(?i)Food.*" RETURN m.name,type(r),n.name
    query_result = graph.run(Query)
    for triple in query_result:
        raw_triple = str(triple).replace('\t', ',')
        extract_service_name =  extract_type_service_name_from_triple(raw_triple)
        services_name.append(extract_service_name)
    logger.debug(f"Services list based type: {services_name}")
    return services_name

# * Get duplicate service name from list 
def get_duplicate_service_name(services_name):
    print("get_duplicate_service_name")
    duplicates = [item for item in set(services_name) if services_name.count(item) > 1]
    if len(duplicates) == 0:
        duplicates = services_name[0:5]
    return duplicates

def get_serving_from_list(audience_list, logger):
    list_size = len(audience_list)
    print("serving length is {0}".format(list_size))
    services_name = []
    if list_size == 1:
        services_name = get_service_serving(audience_list[0], logger=logger)
    if list_size == 2:
        services_name = get_service_serving(audience_list[0], audience_list[1], logger=logger)
    if list_size == 3:
        services_name = get_service_serving(audience_list[0], audience_list[1], audience_list[2], logger=logger)
    if list_size == 4:
        services_name = get_service_serving(audience_list[0], audience_list[1], audience_list[2], audience_list[3], logger=logger)
    if list_size == 5:
        services_name = get_service_serving(audience_list[0], audience_list[1], audience_list[2], audience_list[3], audience_list[4], logger=logger)
    if list_size == 6:
        services_name = get_service_serving(audience_list[0], audience_list[1], audience_list[2], audience_list[3], audience_list[4], audience_list[5], logger=logger)
    if list_size == 7:
        services_name = get_service_serving(audience_list[0], audience_list[1], audience_list[2], audience_list[3], audience_list[4], audience_list[5], audience_list[6], logger=logger)
    return services_name
    

def set_logger(log_file = 'streamlit.log', log_level=logging.DEBUG):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    # record log in log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # record log in console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

def filter_crime_based_zipcode(crime_array: list, zipcode: str):

    """
    filter crime information

    Args:
        crime_array (list): crime's information list
        zipcode (str): crime happen's zipcode

    Returns:
        filter_crime: crime information based on zipcode
    """    

    print('filter_crime')
    filter_crime = []
    index = 0
    for crime in crime_array:
        if crime['zipcode'] == int(zipcode):
            filter_crime.append(crime['info'])
        index = index + 1
    return filter_crime

def get_crimes_summary(crimes_list, st, language = 'en'):

    """
    Get crimes' summary information

    Args:
        crimes_list (list): crime's information list
        st (streamlit): interface pointer
        language (str, optional): translation language. Defaults to 'en'.
    """    

    if len(crimes_list) == 0:
        st.write('There is no crime record in this area.')
        return
    summary_prompt = f"""
Please combine these crime informations, construct corresponding natural language sentences about most common crime type and crime rate per day.
response: {crimes_list}
"""
    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # Updated to use the latest and more advanced model
                    messages=[
                        {"role": "user", "content": summary_prompt}
                    ],
                    temperature=0.2
                )

    crimes_information = response.choices[0].message['content']
    if len(crimes_information) > google_translator_max_char:
        crimes_information = crimes_information[:google_translator_max_char]
    crimes_information_trans = GoogleTranslator(source='auto', target=language).translate(str(crimes_information))
    st.write(crimes_information_trans)


def construct_retriever_question(question_type: str, question_information: str, logger) -> str:
    
    """
    Construct question used by G-Retriever

    Args:
        question_type (str): (time , zipcode, audience)
        question_information (str): information extracted from user query
        logger : python logger
    """    

    question = ''
    if question_type == 'time':
        question = question_information + ' from ? to ?'

    if question_type == 'zipcode':
        question = question_information

    if question_type == 'audience':
        question = question_information

    logger.debug("G-Retriever question: " + question)
    return question


def extract_subgraph_based_on_query(service_type: str, question_type: str, question_information: str, logger):

    """
    Extract subgraph based on query type (time , zipcode, audience) (G-Retriever search)

    Args:
        service_type (str): (food, shelter, mental_health)
        question_type (str): (time , zipcode, audience)
        question_information (str): information extracted from user query
        logger : python logger
    """    

    # 1. embeding question and graph
    if service_type == "Shelter":
        service_type = "shelter"
    if service_type == "Mental Health":
        service_type = "mental_health"
    if service_type == "Food":
        service_type = "food"

    model, tokenizer, device = load_model[embeding_model_name]()
    text2embedding = load_text2embedding[embeding_model_name]

    # 1.1 encode questions
    print('Encoding questions...')
    questions = []
    questions.append(construct_retriever_question(question_type, question_information, logger))
    q_embs = text2embedding(model, tokenizer, device, questions)

    # 2. embeding question and graph
    for index in range(len(q_embs)):
        nodes = pd.read_csv(f'{path_nodes}/' + service_type + '/'+ f'{0}.csv')
        edges = pd.read_csv(f'{path_edges}/' + service_type + '/'+ f'{0}.csv')

        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue

        graph = torch.load(f'{path_graphs}/' + service_type + '/'+ f'{0}.pt')
        q_emb = q_embs[index]
        subg, desc = retrieval_via_pcst(graph, q_emb, nodes, edges, topk=5, topk_e=1, cost_e=0.5)
        logger.debug(desc)
        # open(f'{cached_desc}/{index}.txt', 'w+').write(desc)
    return desc

def create_bnb_config():

    """
    Used to load model, llm model bit config

    Returns:
        bnb_config (class): the bit config class
    """    

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False,
        bnb_8bit_use_double_quant=False
    )

    return bnb_config

def load_llm_model(model_name, bnb_config):

    """
    Used to load model

    Args:
        model_name (string): model's name
        bnb_config (class): bit config class

    Returns:
        model(class): llm model
        tokenizer(class): model tokenizer
    """    

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = 'auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, low_cpu_mem_usage = True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def init_llm_model(model_name : str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"):

    """
    Init llm model

    Args:
        model_name (str, optional): model name. Defaults to "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B".

    Returns:
        model: llm model  
        tokenizer: llm model tokenizer
    """    

    bnb_config = create_bnb_config()
    model, tokenizer = load_llm_model(model_name, bnb_config)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_merged_dir = "./results/deepseek/final_merged_checkpoint"
    os.makedirs(output_merged_dir, exist_ok=True)
    model.save_pretrained(output_merged_dir, safe_serialization=True)

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    tokenizer.save_pretrained(output_merged_dir)
    return model, tokenizer

def ask_model_for_service_extraction(model, question, tokenizer, logger):

    """
    Ask llm model to extract service list based on G-Retriever sub-graph


    Args:
        model (class): llm model
        question (string): G-Retriever sub-graph
        tokenizer (class): model tokenizer
        logger : python logger

    Returns:
        service_list: service name list
    """ 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extraction_instruction = "Can you help me extract a service list based on below nodes, edges, graph information? Only response a service name list.\n"
    # full_conversation = conversation_history + [{"role": "user", "content": combined_query}]
    full_question = extraction_instruction + question
    inputs = tokenizer(full_question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=2550, pad_token_id=tokenizer.eos_token_id)
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # TODO: Create a services list based on model answer
    service_list = []
    clean_text = model_answer.split("</think>")[-1]
    matches = re.findall(r'\d+\.\s*(.*)', clean_text)
    for service in matches:
        service_list.append(service)

    return service_list

def extract_google_rating_from_triple(triple):

    """
    extract google rating from triple

    Args:
        triple (string): triple string

    Returns:
        string: extracted service rating
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

    # start_quotation = triple.find('"')
    target_index = triple.find('\'ratingValue\'', 1)
    extract_rating = triple[target_index : -1]
    return extract_rating 

def google_order(service_list : list):

    """
    Google Rating Order

    Returns:
        list: reordered service list
    """    

    sorted_places = sorted(service_list, key=lambda x: x[1], reverse=True)
    return sorted_places

def get_services_type(service_type, logger, relation='service type'):
    print('get_services_type')
    services_name = []
    graph = Graph(
            "bolt://localhost:7687", 
            auth=("neo4j", "123456789")
        )
    Query = 'MATCH (m:node)-[r]->(n:node) where type(r)=~\".*(?i){0}.*\" and n.name=~\".*(?i){1}.*\" RETURN m.name,type(r),n.name'.format(relation, service_type)
    # MATCH (m:node)-[r]-(n:node) where type(r)=~".*(?i)service type.*" and n.name=~".*(?i)Food.*" RETURN m.name,type(r),n.name
    query_result = graph.run(Query)
    for triple in query_result:
        raw_triple = str(triple).replace('\t', ',')
        extract_service_name =  extract_type_service_name_from_triple(raw_triple)
        services_name.append(extract_service_name)
    logger.debug(f"Services list based type: {services_name}")
    return services_name

def extract_type_service_name_from_triple(triple):

    """
    extract service name from triple

    Args:
        triple (string): triple string

    Returns:
        string: extracted service name
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

    # start_quotation = triple.find('"')
    target_index = triple.find('\'service type\'', 1)
    extract_service_name = triple[0:target_index - 1]
    if extract_service_name[0] == '\'':
        extract_service_name = extract_service_name.rstrip('\'')
        extract_service_name = extract_service_name.lstrip('\'')
    else:
        extract_service_name = extract_service_name.rstrip('\"')
        extract_service_name = extract_service_name.lstrip('\"')
    return extract_service_name  

def extract_zipcode_from_triple(triple):
    target_index = triple.find('\'postalCode\'', 1)
    extract_zipcode= triple[target_index + 15 : -1]
    return extract_zipcode.replace('"', '')

def get_zipcode(service_list : list, logger):
    relation = 'postalCode'
    service_zipcode = []
    graph = Graph(
            "bolt://localhost:7687", 
            auth=("neo4j", "123456789")
        )
    for service in service_list:
        Query = 'MATCH (m:node {{name:"{1}"}})-[r]->(n:node) where type(r)=~\".*(?i){0}.*\" RETURN m.name,type(r),n.name'.format(relation, service[0])
        query_result = graph.run(Query)
        #! Only need one return result, sometime store two zipcode info or same short name service
        for triple in query_result:
            raw_triple = str(triple).replace('\t', ',')
            extract_zipcode = extract_zipcode_from_triple(raw_triple)
            service_zipcode.append([service[0], service[1], int(extract_zipcode)])
            break

    logger.debug(f"Services list based zipcode: {service_zipcode}")

    return service_zipcode

def get_location(service_list : list, logger):
    longitude = 'longitude'
    latitude = 'latitude'
    service_location = []
    graph = Graph(
            "bolt://localhost:7687", 
            auth=("neo4j", "123456789")
        )

    for service in service_list:
        latitude_Query = 'MATCH (m:node)-[r]->(n:node) where type(r)=~\".*(?i){0}.*\" and m.name=~\".*(?i){1}.*\" RETURN m.name,type(r),n.name'.format(latitude, service)
        query_result = graph.run(latitude_Query)

def extract_service_list_from_string(response_service_list:str, service_list:list):
    result = []
    chunks = response_service_list[2:-2].split("], [")
    for chunk in chunks:
        # Find the last comma in the chunk (splits name and rating)
        idx = chunk.rfind(',')
        name = chunk[:idx].strip().strip("'")
        rating = float(chunk[idx+1:].strip())
        result.append([name, rating])

    for response_service in result:
        for service in service_list:
            if response_service[0] == service[0]:
                response_service.append(service[2])
                response_service.append(service[3])
    return result


def distance_order(service_list : list, zipcode, logger):
    service_zipcode = get_zipcode(service_list, logger)
    question_prompt = f'''My location zipcode is {zipcode}, and there are some services and its google rating and zipcode.{service_zipcode}. Please help me reorder these services based on distance from my location to service's location.
    And only return the reordered list of serive name and google rating. Such as: [['The West Philly Bunny Hop', 0.0], ["St. Barbara's Roman Catholic Church", 4.4]]'''
    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",  # Updated to use the latest and more advanced model
                        messages=[
                            {"role": "user", "content": question_prompt}
                        ],
                        temperature=0.2
                    )
    response_service_list = response.choices[0].message['content']
    final_list = extract_service_list_from_string(response_service_list, service_list)
    return final_list


def order_service(service_List : list, logger, zipcode, order_method = GOOGLE_RATING):

    """
    order service list

    Args:
        service_List (list): service list([service name, google rating])
        logger (class): log class
        order_method (int, optional): order method. Defaults to GOOGLE_RATING.

    Returns:
        list: reordered service list
    """    

    if order_method == GOOGLE_RATING:
        sorted_service = google_order(service_List)
        logger.debug(f"Google Rating Sorted Service List: {sorted_service} ")

    if order_method == DISTANCE_ORDER:
        sorted_service = distance_order(service_List, zipcode, logger)
        logger.debug(f"Distance Sorted Service List: {sorted_service} ")

    return sorted_service