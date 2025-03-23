import pandas as pd
from neo4j import GraphDatabase
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import re
from typing import List
import json
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
from llm_prompts import *
from langchain.chat_models import ChatOpenAI
import cohere
from nltk.stem import WordNetLemmatizer

llm = None
def initialise_mixtral(temperature, max_tokens):
    print("--Initialising Mixtral")
    global llm
    llm = OllamaLLM(model="mixtral:8x7b")

def initialise_llama3(temperature, max_tokens):
    print("--Initialising Llama3")
    global llm
    llm = OllamaLLM(model="llama3:70b-instruct")

def initialise_gpt(temperature, max_tokens, opeani_api_keys, model_name):
    print("--Initialising GPT")
    global llm
    llm = ChatOpenAI(
    model=model_name,
    api_key=opeani_api_keys,
    temperature=temperature,
    max_tokens=max_tokens,
    request_timeout=30,
    max_retries=3,
    timeout=60 * 3,
)


session = None
def initialise_neo4j():
    print("Connecting to Neo4j")
    global session
    uri = "bolt://54.144.18.207"
    username = "neo4j"
    password = "bins-advertisements-contributions"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()

def load_kg_in_neo4j(read_dataset):
    session.run("MATCH (n) DETACH DELETE n")

    # read triples
    if read_dataset == "MedicationQA":
        df = pd.read_csv('../data/MedicationQA/medlineplus_77_triples.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])
    elif read_dataset == "GenMedGPT":
        df = pd.read_csv('../data/DisTreatKG/DisTreatKG_train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])

    for index, row in df.iterrows():
        head_name = row['head']
        tail_name = row['tail']
        relation_name = row['relation']

        query = (
            "MERGE (h:Entity { name: $head_name }) "
            "MERGE (t:Entity { name: $tail_name }) "
            "MERGE (h)-[r:`" + relation_name + "`]->(t)"
        )
        try:
            session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
        except:
            continue

def extract_question_and_keywords(input):
    re1 = r"<CLS>(.*?)<SEP>"
    input = input.replace("\n","")
    input = input.replace("<OOS>","<EOS>")
    input = input.replace(":","") + "<END>"
    input_text = re.findall(re1,input)

    re2 = r'The extracted entities are (.*?)<END>'
    re3 = r"The extracted entity is (.*?)<END>"

    question_kg = re.findall(re2,input) 
    if len(question_kg) == 0:
        question_kg = re.findall(re3,input) 
        if len(question_kg) == 0:
            print("<Warning> no entities found", input)
    question_kg = question_kg[0].replace("<END>","").replace("<EOS>","")
    question_kg = question_kg.replace("\n","")
    question_kg = question_kg.split(", ")
    
    return input_text, question_kg

def extract_answer(output):
    re1 = r"<CLS>(.*?)<SEP>"
    output = output.replace("\n","")
    output = output.replace("<OOS>","<EOS>")
    output = output.replace(":","") + "<END>"
    output_text = re.findall(re1,output)
    return output_text

def read_genmedgpt_dataset():
    print("Reading GenMedGPT dataset")
    with open("../data/GenMedGPT-5k/GenMedGPT_test_data_with_NER.json", encoding='utf-8') as f:
        data = f.read()
        data = json.loads(data)
        # print(data)
        question = []
        answers = []
        topic_entities = []

        for line in enumerate(data):
            input = line[1]["qustion_output"]
            input_text, question_kg = extract_question_and_keywords(input)
            question.append(input_text[0])
            
            topic_entities.append(question_kg)

            output = line[1]["answer_output"]
            output_text = extract_answer(output)
            answers.append(output_text[0])

    dataset = pd.DataFrame({
        'Question': question, 
        'Answer': answers, 
        'Topic_Entities': topic_entities})
    return dataset

def read_medqa_dataset():
    print("Reading MedQA dataset")
    file = '../data/MedicationQA/MedicationQA_medlineplus_93.xlsx'
    # file = '../data/MedicationQA/MedicationQA_medlineplus_bottom20.xlsx'
    dataset = pd.read_excel(file)
    return dataset

def find_entity(entity_id):
    # print("Finding entitiy in Kg")
    words = entity_id.split()
    re_pattern = r"(?i).*(" + "|".join(re.escape(word) for word in words) + r").*"

    find_entity_query = """MATCH (entity) WHERE toLower(entity.name) =~ $re_pattern RETURN entity.name AS entity_found"""
    result = session.run(find_entity_query, parameters={"re_pattern":re_pattern})
    return [record["entity_found"] for record in result]

mpnet_model = None
def initialise_mpnet_model():
    global mpnet_model
    mpnet_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def mpnet_encode(texts):
    return mpnet_model.encode(texts, convert_to_tensor=False)
    
def compute_similarity(embedding1, embedding2):
    return util.cos_sim(embedding1, embedding2).item()

def filter_similar_entities(question_entity, kg_entities):
    # print("Finding similar entity threshold")
    question_embedding = mpnet_encode([question_entity])[0]
    kg_embeddings = mpnet_encode(kg_entities)
    match_80 = []
    match_60 = []
    match_50 = []
    match_kg = []
    
    for kg_entity, kg_embedding in zip(kg_entities, kg_embeddings):
        similarity = compute_similarity(question_embedding, kg_embedding)
        if similarity >= 0.8:
            match_80.append(kg_entity)
        elif similarity >= 0.6:
            match_60.append(kg_entity)
        elif similarity >= 0.5:
            match_50.append(kg_entity)

    if len(match_80) != 0:
        match_kg = match_80
    elif len(match_60) != 0:
        match_kg = match_60
    elif len(match_50) != 0:
        match_kg = match_50
    return match_kg

lemmatizer = WordNetLemmatizer()
def find_exact_entity(entity_id):
    word_list = entity_id.split()
    lemma_words = [lemmatizer.lemmatize(word) for word in word_list]
    lemmatized_entity = '_'.join(lemma_words)
    
    re_pattern = r"(?i)\b" + re.escape(lemmatized_entity) + r"\b"

    find_entity_query = """MATCH (entity) WHERE toLower(entity.name) =~ $re_pattern RETURN entity.name AS entity_found"""
    result = session.run(find_entity_query, parameters={"re_pattern":re_pattern})
    return [record["entity_found"] for record in result]

def calculate_shortest_path(start_entity_name, end_entity_name):
    
    query = """
            MATCH (start_entity:Entity {name: $start_entity_name}), (end_entity:Entity {name: $end_entity_name})
            MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity))
            UNWIND relationships(p) AS r
            RETURN startNode(r) AS subject, type(r) AS predicate, endNode(r) AS object
            """        
    result = session.run(query, start_entity_name=start_entity_name, end_entity_name=end_entity_name)
    return result.data()

def stitch_shortest_path_triples(triples):
    triple_dict = {}
    for subj, pred, obj in triples:
        if subj not in triple_dict:
            triple_dict[subj] = []
        triple_dict[subj].append((pred, obj))
    
    visited = set()
    stitched_triples = set()
    for subj in triple_dict:
        for pred, obj in triple_dict[subj]:
            if obj in triple_dict and obj not in visited:
                for next_pred, next_obj in triple_dict[obj]:
                    stitched_triples.add((subj, pred, obj, next_pred, next_obj))
                    visited.add(obj)
            elif obj in triple_dict:
                stitched_triples.add((subj, pred, obj))
    
    return list(stitched_triples)

def find_shortest_paths_between_list(match_kg):
    all_triples = []
    for i in range(len(match_kg) - 1):
        for j in range(i + 1, len(match_kg)):
            start_entity_name = match_kg[i]
            end_entity_name = match_kg[j]
            path_info = calculate_shortest_path(start_entity_name, end_entity_name)
            for path in path_info:        
                subject = path['subject']['name']
                predicate = path['predicate']
                object = path['object']['name']
                all_triples.append((subject, predicate, object))
               
    paths_found = stitch_shortest_path_triples(all_triples)
            
    return paths_found

def find_neighbors_paths(entity_name: str) -> List[List[str]]:
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    for record in result:
        rel_type = record["relationship_type"]
        
        neighbors = record["neighbor_entities"]
        
        neighbor_list.append([entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in neighbors])
                                ])
    return neighbor_list

def rerank_kg_output(query, triples, top_n):
    rerank_model = CrossEncoder("ncbi/MedCPT-Cross-Encoder")
    if not triples:
        return [] 
    pairs = [[query, trip] for trip in triples]
    try:
        scores = rerank_model.predict(pairs)
    except:
        scores = np.zeros(len(triples))
    
    cross_encoded_scores = list(zip(scores, triples))
    cross_encoded_scores.sort(key=lambda x: x[0], reverse=True)
    rerank_rels = cross_encoded_scores[:top_n]
    
    reranked_results = []
    for score, path in rerank_rels:
        reranked_results.append(path)
    return reranked_results

def rerank_kg_output_cohere(query, triples, api_key, top_n):
    co = cohere.ClientV2(api_key)
    if not triples:
        return [] 
    try:
        results = co.rerank(query=query, documents=triples, top_n=top_n, model='rerank-english-v3.0', return_documents=True)
    except:
        results = np.zeros(len(triples))
    reranked_results = []
    try:
        for r in results.results:
            reranked_results.append(r.document.text)
        return reranked_results
    except: 
        return reranked_results

def final_answer_with_kg_mx(text, kg_output, read_dataset):

    if read_dataset == "MedicationQA":
        prompt = ChatPromptTemplate.from_template(medqa_final_answer_template_with_kg_mx)
    elif read_dataset == "GenMedGPT":
        prompt = ChatPromptTemplate.from_template(medqa_final_answer_template_with_kg_mx)

    llm_chain = prompt | llm
    
    output_all = llm_chain.invoke({"text": text, "kg_output": kg_output})
    
    return output_all.strip()

def final_answer_with_kg_l3(text, kg_output, read_dataset):
    
    if read_dataset == "MedicationQA":
        prompt = ChatPromptTemplate.from_template(medqa_final_answer_template_with_kg_l3)
    elif read_dataset == "GenMedGPT":
        prompt = ChatPromptTemplate.from_template(genmed_final_answer_template_with_kg_l3)

    llm_chain = prompt | llm
    
    output_all = llm_chain.invoke({"text": text, "kg_output": kg_output})
    
    return output_all.strip()

def final_answer_with_kg_gpt(text, kg_output, read_dataset):

    if read_dataset == "MedicationQA":
        prompt = ChatPromptTemplate.from_template(medqa_final_answer_template_with_kg_l3)
    elif read_dataset == "GenMedGPT":
        prompt = ChatPromptTemplate.from_template(genmed_final_answer_template_with_kg_l3)

    llm_chain = prompt | llm
    
    output_all = llm_chain.invoke({"text": text, "kg_output": kg_output})
    
    return output_all.content

def final_answer_only_llm_mx(text):

    prompt = ChatPromptTemplate.from_template(final_answer_template_only_llm_mx)

    llm_chain = prompt | llm
    
    output_all = llm_chain.invoke({"text": text})
    
    return output_all.strip()

def final_answer_only_llm_l3(text):
    
    prompt = ChatPromptTemplate.from_template(final_answer_template_only_llm_l3)

    llm_chain = prompt | llm
    
    output_all = llm_chain.invoke({"text": text})
    
    return output_all.strip()

def final_answer_only_llm_gpt(text):
    prompt = ChatPromptTemplate.from_template(final_answer_template_only_llm_l3)

    llm_chain = prompt | llm
    
    output_all = llm_chain.invoke({"text": text})
    
    return output_all.content