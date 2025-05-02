## Data-preprocess
#### crime data
crime record only have geomatric information, you need to execute below code to get zipcode of crime and filter recent three days' crime record.
**Warning:**  If there are no new update crime records, you don't need to execute this step.
```python
python filter_crime.py
```

## Preprocesss 

#### Setup the Neo4j 

1. Make sure you have already start Neo4j service in your server 
2. Execute below code to store graph triples information into Neo4j
    ```python
    python store_neo4j.py
    ```

#### Prepare for g-retriever graph

1. Execute below code to prepare graph used for g-retriever
    ```python
    python retriever-preprocess.py
    ```


## Start

Make sure you have already has 'streamlit' module to start interface.
```python

streamlit run DREAM-KG_Chatbot.py
```