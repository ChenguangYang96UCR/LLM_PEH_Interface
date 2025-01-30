import re
import streamlit as st
import openai
from py2neo import Graph
from deep_translator import GoogleTranslator
import logging

google_translator_max_char = 5000

# * Extract weekday from string
def extract_weekday(input_string):
    # extract weekday from string
    weekday_match = re.search(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", input_string)
    return weekday_match.group(1) if weekday_match else None

# * Get service open time from graph
def get_services_time(weekday, relation='xmlschema11-2#time'):
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
    triples = []
    if key_word == '':
        print('Error: search_node_by_keyword input \'key_word\' is nothing')
        return triples
    graph = Graph(
            "bolt://localhost:7687", 
            auth=("neo4j", "123456789")
        )
    if relation == '':
        Query = 'MATCH (m:node)-[r]-(n:node) where m.name=~\".*(?i){0}.*\" or n.name=~\".*(?i){0}.*\" RETURN m.name,type(r),n.name'.format(key_word)
        query_result = graph.run(Query)
        for triple in query_result:
            triples.append(str(triple).replace('\t', ','))
    else:
        Query = 'MATCH (m:node)-[r]-(n:node) where m.name=~\".*(?i){0}.*\" or n.name=~\".*(?i){0}.*\" RETURN m.name,type(r),n.name'.format(key_word, relation)
        query_result = graph.run(Query)
        for triple in query_result:
            triples.append(str(triple).replace('\t', ','))
    return triples

# * Get and combine services' information 
@st.cache_resource(show_spinner=False)
def getQuestion_answer(Service_list, st, language = 'en'):
    print("getQuestion_answer")
    all_triples = []    
    all_information = []
    for Service in Service_list:
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
Please combine these response, construct corresponding natural language sentences about introduction and suggestion.
response: {all_response}
"""
        response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",  # Updated to use the latest and more advanced model
                        messages=[
                            {"role": "user", "content": combine_prompt}
                        ],
                        temperature=0.2
                    )

        st.write(f"**{Service[0]}**")
        google_rating = f"**Google Rating:{Service[1]}**"
        google_rating_trans = GoogleTranslator(source='auto', target=language).translate(str(google_rating))
        st.write(google_rating_trans, "\n")
        service_information = response.choices[0].message['content']
        if len(service_information) > google_translator_max_char:
            service_information = service_information[:google_translator_max_char]
        service_information_trans = GoogleTranslator(source='auto', target=language).translate(str(service_information))
        st.write(service_information_trans)

    # st.write(f"###{Service[0]} Google Rating:{Service[1]}", "\n", response.choices[0].message['content'])
    print("Finished GetQuestion_Answer!")
    return all_information

def extract_service_name_from_triple(triple):
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
    logger.debug(Query)
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
        logger.debug(Query)
    if len(args) == 2:
        Query = 'MATCH (m:node)-[r]->(n:node) where type(r)=~\".*(?i){0}.*\" and (n.name=~\".*(?i){1}.*\" or n.name=~\".*(?i){2}.*\") RETURN m.name,type(r),n.name'.format(relation, args[0], args[1])
        logger.debug(Query)
    if len(args) == 3:
        Query = 'MATCH (m:node)-[r]->(n:node) where type(r)=~\".*(?i){0}.*\" and (n.name=~\".*(?i){1}.*\" or n.name=~\".*(?i){2}.*\" or n.name=~\".*(?i){3}.*\") RETURN m.name,type(r),n.name'.format(relation, args[0], args[1], args[2])
        logger.debug(Query)
    if len(args) == 4:
        Query = 'MATCH (m:node)-[r]->(n:node) where type(r)=~\".*(?i){0}.*\" and (n.name=~\".*(?i){1}.*\" or n.name=~\".*(?i){2}.*\" or n.name=~\".*(?i){3}.*\" or n.name=~\".*(?i){4}.*\") RETURN m.name,type(r),n.name'.format(relation, args[0], args[1], args[2], args[3])
        logger.debug(Query)

    query_result = graph.run(Query)
    for triple in query_result:
        raw_triple = str(triple).replace('\t', ',')
        extract_service_name =  extract_service_name_from_triple(raw_triple)
        services_name.append(extract_service_name)
    unique_service = list(set(services_name))
    return unique_service

# * Get duplicate service name from list 
def get_duplicate_service_name(services_name):
    print("get_duplicate_service_name")
    duplicates = [item for item in set(services_name) if services_name.count(item) > 1]
    if len(duplicates) == 0:
        duplicates = services_name
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

def filter_crime_based_zipcode(crime_array, zipcode):
    print('filter_crime')
    filter_crime = []
    index = 0
    for crime in crime_array:
        if crime['zipcode'] == int(zipcode):
            filter_crime.append(crime['info'])
        index = index + 1
    return filter_crime

def get_crimes_summary(crimes_list, st, language = 'en'):
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