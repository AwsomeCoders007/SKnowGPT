import pandas as pd
from neo4j import GraphDatabase
import numpy as np
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import pickle
import re
from typing import List
import csv
import json
import torch
import time
import itertools

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results

def find_shortest_path(start_entity_name, end_entity_name,candidate_list):
    global exist_entity
    exist_entity = {}
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)
           
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_"," ")
                
                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_"," ")
                    path_str += "->" + relations[i] + "->"
            
            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}
            
        if len(paths) > 5:        
            paths = sorted(paths, key=len)[:5]

        return paths,exist_entity

def get_entity_neighbors(entity_name: str,disease_flag) -> List[List[str]]:
    disease = []
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
        
        if disease_flag == 1 and rel_type == 'has_symptom':
            continue

        neighbors = record["neighbor_entities"]
        
        if "disease" in rel_type.replace("_"," "):
            disease.extend(neighbors)

        else:
            neighbor_list.append([entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in neighbors])
                                ])
    
    return neighbor_list,disease

def prompt_path_finding_mx(text):
    template = """<s> [INST]
    Below knowledge graph path information is provided below in entity->relationship->entity format. Convert this information to natural language, respectively. \
    Use single quotation marks for entity name and relation name. 
    {Path}
    [/INST]
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["Path"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    response_of_KG_path = llm_chain.run(text)
    return response_of_KG_path

def prompt_neighbor_mx(text):
    template = """<s> [INST]
    Below knowledge graph neighbor information is provided in entity->relationship->entity format. Convert each line to a sentence, transforming it into natural language. \
    Use single quotation marks for entity names and relation names. \
    Ensure that all information from each line is integrated into a single sentence. 
    {neighbor}
    [/INST]
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["neighbor"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    response_of_KG_neighbor = llm_chain.run(text)

    return response_of_KG_neighbor

def prune_kg_mx(kg_input, question):
    template = """<s> [INST]
    Remove redundant or irrelevant text from the path based and neighbor based information provided in the additional information.
    Do not modify the sentences in any way.\
    Do not delete the tests or treatments for the disease or symptoms mentioned in the question.\
    Example:
    Patient Input:
    Doctor, I have been experiencing frequent stomach pain, nausea and vomiting. I think I might have gastritis.
    
    Additional information:
    1. 'Sharp abdominal pain' is a possible disease of 'Gastritis'.
    2. 'Gastritis' has the symptom of 'Nausea'.
    3. 'Nausea' is a possible disease that can cause 'Nausea'.
    4. 'Gastritis', which causes 'Nausea', also has the symptom of 'Vomiting'.
    5. 'Vomiting' is a possible disease that can result from 'Vomiting'.
    6. Finally, 'Gastritis', which is associated with both 'Nausea' and 'Vomiting', is a possible disease that can cause 'Vomiting'.
    1. Gastritis is associated with the symptoms Sharp abdominal pain, Vomiting, Nausea, Burning abdominal pain, Sharp chest pain, Upper abdominal pain, Diarrhea, Fever, Headache, Heartburn, Vomiting blood, and Regurgitation.
    2. In the case of gastritis, medical tests such as Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel are required.
    3. To treat gastritis, medications like Amylases, Aluminum Hydroxide (M.A.H.), Benzocaine Topical, Famotidine, Pantoprazole, Ranitidine, Sucralfate (Carafate), Cimetidine, and Bismuth Subsalicylate (Pepto-Bismol) may be needed.
    4. For diagnosing gastritis, medical tests including Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel are necessary.
    5. In managing gastritis, patients might be prescribed medications such as Amylases, Aluminum Hydroxide (M.A.H.), Benzocaine Topical, Famotidine, Pantoprazole, Ranitidine, Sucralfate (Carafate), Cimetidine, and Bismuth Subsalicylate (Pepto-Bismol).
    
    Output:
    1. 'Sharp abdominal pain' is a possible symptom of 'Gastritis'.
    2. 'Gastritis' has the symptom of 'Nausea'.
    4. 'Gastritis', which causes 'Nausea', also has the symptom of 'Vomiting'.
    1. Gastritis is associated with the symptoms Sharp abdominal pain, Nausea, and Vomiting.
    2. In the case of gastritis, medical tests such as Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel are required.
    3. To treat gastritis, medications like Amylases, Aluminum Hydroxide (M.A.H.), Benzocaine Topical, Famotidine, Pantoprazole, Ranitidine, Sucralfate (Carafate), Cimetidine, and Bismuth Subsalicylate (Pepto-Bismol) may be needed.

    **Question:**
    {question}

    **Additional information:** 
    {kg_input}
    [/INST]
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["question", "kg_input"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response_of_KG = llm_chain.run(kg_input=kg_input,question=question)
    return response_of_KG

def final_answer_without_dtt_no_cot_no_kg_mx(text):
    template = """<s>[INST]
    You are an excellent AI doctor, and you can provide medical advice to patient's query.\
    If a question does not make sense or is not factually coherent, explain why instead of providing incorrect information. \
    If you don't know the answer to a question, refrain from sharing false information. Provide consise answers. Think step by step.
    [/INST]
    **Patient Input:**
    {text}
    </s>
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["text"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    output_all = llm_chain.run(text=text)
    
    return output_all

def final_answer_without_dtt_no_cot_with_kg_mx(text, response_of_KG_list_path, response_of_KG_neighbor):
    
    template = """<s>[INST]
    You are an excellent AI doctor, and you can provide medical advice to patient's query based on the provided medical knowledge information.\
    If a question does not make sense or is not factually coherent, explain why instead of providing incorrect information. \
    If you don't know the answer to a question, refrain from sharing false information. Provide consise answers. Think step by step.

    **Patient Input:**
    {text}
    **Additional medical knowledge information:** 
    {response_of_KG_list_path}
    {response_of_KG_neighbor}
    [/INST]

    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["response_of_KG_list_path","response_of_KG_neighbor", "text"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    # print(prompt.format(text=text,response_of_KG_list_path=response_of_KG_list_path,response_of_KG_neighbor=response_of_KG_neighbor))
    output_all = llm_chain.run(text=text,response_of_KG_list_path=response_of_KG_list_path,response_of_KG_neighbor=response_of_KG_neighbor)
    
    return output_all

def final_answer_without_dtt_cot1_full_kg_mx(text, response_of_KG_list_path, response_of_KG_neighbor):
    #cot 5-2-24 final ans v5
    template = """<s>[INST]
    You are an excellent AI doctor, and you can provide medical advice to patient's query based on the provided medical knowledge information.\
    If a question does not make sense or is not factually coherent, explain why instead of providing incorrect information. If you don't know the answer to a question, refrain from sharing false information. Provide consise answers. Think step by step.
    Example:
    Patient Input:
    Doctor, I have been experiencing frequent stomach pain, nausea and vomiting. I think I might have gastritis.
    
    Additional medical knowledge information:
    Path-based Evidence 1: 'Sharp abdominal pain' is a possible disease of 'Gastritis'.
    Path-based Evidence 2: 'Gastritis' has the symptom of 'Nausea'.
    Path-based Evidence 3: 'Nausea' is a possible disease that can be associated with 'Gastritis'.
    Path-based Evidence 4: 'Gastritis' has the symptom of 'Vomiting'.
    Path-based Evidence 5: 'Vomiting' is a possible disease that can be associated with 'Gastritis'.
    Neighbor-based Evidence 1: Gastritis is associated with the symptoms Sharp abdominal pain, Vomiting, Nausea, Burning abdominal pain, Sharp chest pain, Upper abdominal pain, Diarrhea, Fever, Headache, Heartburn, Vomiting blood, and Regurgitation.
    Neighbor-based Evidence 2: In the case of gastritis, medical tests such as Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel are required.
    Neighbor-based Evidence 3: To manage gastritis, medications like Amylases, Aluminum Hydroxide (M.A.H.), Benzocaine Topical, Famotidine, Pantoprazole, Ranitidine, Sucralfate (Carafate), Cimetidine, and Bismuth Subsalicylate (Pepto-Bismol) may be needed.
    Neighbor-based Evidence 4: For diagnosing gastritis, medical tests including Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel are necessary.
    Neighbor-based Evidence 5: In treating gastritis, patients may be prescribed medications such as Amylases, Aluminum Hydroxide (M.A.H.), Benzocaine Topical, Famotidine, Pantoprazole, Ranitidine, Sucralfate (Carafate), Cimetidine, and Bismuth Subsalicylate (Pepto-Bismol).
    
    Output:
    In order to confirm whether you have gastritis or not, I would like to recommend a few medical tests. We need to conduct Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel. 
    
    **Patient Input:**
    {text}
    **Additional medical knowledge information:** 
    {response_of_KG_list_path}
    {response_of_KG_neighbor}
    [/INST]

    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["response_of_KG_list_path","response_of_KG_neighbor", "text"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    # print(prompt.format(text=text,response_of_KG_list_path=response_of_KG_list_path,response_of_KG_neighbor=response_of_KG_neighbor))
    output_all = llm_chain.run(text=text,response_of_KG_list_path=response_of_KG_list_path,response_of_KG_neighbor=response_of_KG_neighbor)
    
    return output_all

def final_answer_without_dtt_cot1_no_kg_mx(text):
    #cot 5-2-24 final ans v5
    template = """<s>[INST]
    You are an excellent AI doctor, and you can provide medical advice to patient's query.\
    If a question does not make sense or is not factually coherent, explain why instead of providing incorrect information. If you don't know the answer to a question, refrain from sharing false information. Provide consise answers. Think step by step.
    Example:
    Patient Input:
    Doctor, I have been experiencing frequent stomach pain, nausea and vomiting. I think I might have gastritis. 
    Output:
    In order to confirm whether you have gastritis or not, I would like to recommend a few medical tests. We need to conduct Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel. 
    
    **Patient Input:**
    {text}
    [/INST]
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["text"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    output_all = llm_chain.run(text=text)
    
    return output_all

def final_answer_with_dtt_no_cot_with_kg_mx(text, response_of_KG_list_path, response_of_KG_neighbor):
    
    template = """<s>[INST]
    You are an excellent AI doctor, and you can provide medical advice to patient's query based on the provided medical knowledge information.\
    What disease does the patient have? What medical tests should patient take to confirm the diagnosis? What are the recommended medications for the suspected disease?\
    If a question does not make sense or is not factually coherent, explain why instead of providing incorrect information. \
    If you don't know the answer to a question, refrain from sharing false information. Provide consise answers. Think step by step.

    **Patient Input:**
    {text}
    **Additional medical knowledge information:** 
    {response_of_KG_list_path}
    {response_of_KG_neighbor}
    [/INST]

    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["response_of_KG_list_path","response_of_KG_neighbor", "text"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    # print(prompt.format(text=text,response_of_KG_list_path=response_of_KG_list_path,response_of_KG_neighbor=response_of_KG_neighbor))
    output_all = llm_chain.run(text=text,response_of_KG_list_path=response_of_KG_list_path,response_of_KG_neighbor=response_of_KG_neighbor)
    
    return output_all

def final_answer_with_dtt_cot1_with_kg_mx(text, response_of_KG_list_path, response_of_KG_neighbor):
    template = """<s>[INST]
    You are an excellent AI doctor, and you can provide medical advice to patient's query based on the provided medical knowledge information.\
    What disease does the patient have? What medical tests should patient take to confirm the diagnosis? What are the recommended medications for the suspected disease?\
    If a question does not make sense or is not factually coherent, explain why instead of providing incorrect information. If you don't know the answer to a question, refrain from sharing false information. Provide consise answers. Think step by step.
    Example:
    Patient Input:
    Doctor, I have been experiencing frequent stomach pain, nausea and vomiting. I think I might have gastritis.
    
    Additional medical knowledge information:
    Path-based Evidence 1: 'Sharp abdominal pain' is a possible disease of 'Gastritis'.
    Path-based Evidence 2: 'Gastritis' has the symptom of 'Nausea'.
    Path-based Evidence 3: 'Nausea' is a possible disease that can be associated with 'Gastritis'.
    Path-based Evidence 4: 'Gastritis' has the symptom of 'Vomiting'.
    Path-based Evidence 5: 'Vomiting' is a possible disease that can be associated with 'Gastritis'.
    Neighbor-based Evidence 1: Gastritis is associated with the symptoms Sharp abdominal pain, Vomiting, Nausea, Burning abdominal pain, Sharp chest pain, Upper abdominal pain, Diarrhea, Fever, Headache, Heartburn, Vomiting blood, and Regurgitation.
    Neighbor-based Evidence 2: In the case of gastritis, medical tests such as Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel are required.
    Neighbor-based Evidence 3: To manage gastritis, medications like Amylases, Aluminum Hydroxide (M.A.H.), Benzocaine Topical, Famotidine, Pantoprazole, Ranitidine, Sucralfate (Carafate), Cimetidine, and Bismuth Subsalicylate (Pepto-Bismol) may be needed.
    Neighbor-based Evidence 4: For diagnosing gastritis, medical tests including Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel are necessary.
    Neighbor-based Evidence 5: In treating gastritis, patients may be prescribed medications such as Amylases, Aluminum Hydroxide (M.A.H.), Benzocaine Topical, Famotidine, Pantoprazole, Ranitidine, Sucralfate (Carafate), Cimetidine, and Bismuth Subsalicylate (Pepto-Bismol).
    
    Output:
    In order to confirm whether you have gastritis or not, I would like to recommend a few medical tests. We need to conduct Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel. 
    
    **Patient Input:**
    {text}
    **Additional medical knowledge information:** 
    {response_of_KG_list_path}
    {response_of_KG_neighbor}
    [/INST]

    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["response_of_KG_list_path","response_of_KG_neighbor", "text"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    # print(prompt.format(text=text,response_of_KG_list_path=response_of_KG_list_path,response_of_KG_neighbor=response_of_KG_neighbor))
    output_all = llm_chain.run(text=text,response_of_KG_list_path=response_of_KG_list_path,response_of_KG_neighbor=response_of_KG_neighbor)
    
    return output_all

def final_answer_with_dtt_cot1_with_pruned_kg_mx(text, pruned_kg_outputs):
    template = """<s>[INST]
    You are an excellent AI doctor, and you can provide medical advice to patient's query only based on the provided medical knowledge information.\
    What disease does the patient have? What medical tests should patient take to confirm the diagnosis? What are the recommended medications for the suspected disease?\
    If you don't know the answer to a question, refrain from sharing false information. Provide consice answers. Think step by step.
    Example:
    Patient Input:
    Doctor, I have been experiencing frequent stomach pain, nausea and vomiting. I think I might have gastritis.
    
    Additional medical knowledge information:
    Path-based Evidence 1: 'Sharp abdominal pain' is a possible disease of 'Gastritis'.
    Path-based Evidence 2: 'Gastritis' has the symptom of 'Nausea'.
    Path-based Evidence 4: 'Gastritis', which causes 'Nausea', also has the symptom of 'Vomiting'.
    Neighbor-based Evidence 1: Gastritis is associated with the symptoms Sharp abdominal pain, Vomiting, Nausea.
    Neighbor-based Evidence 2: In the case of gastritis, medical tests such as Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel are required.
    Neighbor-based Evidence 3: To manage gastritis, medications like Amylases, Aluminum Hydroxide (M.A.H.), Benzocaine Topical, Famotidine, Pantoprazole, Ranitidine, Sucralfate (Carafate), Cimetidine, and Bismuth Subsalicylate (Pepto-Bismol) may be needed.
    
    Output:
    In order to confirm whether you have gastritis or not, I would like to recommend a few medical tests. We need to conduct Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel. 
    
    **Patient Input:**
    {text}
    **Additional medical knowledge information:** 
    {pruned_kg_outputs}
    [/INST]

    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["pruned_kg_outputs", "text"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    output_all = llm_chain.run(text=text,pruned_kg_outputs=pruned_kg_outputs)
    
    return output_all

def final_answer_without_dtt_cot1_with_pruned_kg_mx(text, pruned_kg_outputs):
    
    template = """<s>[INST]
    You are an excellent AI doctor, and you can provide medical advice to patient's query only based on the provided medical knowledge information.\
    If you don't know the answer to a question, refrain from sharing false information. Provide consise answers. Think step by step.
    Example:
    Patient Input:
    Doctor, I have been experiencing frequent stomach pain, nausea and vomiting. I think I might have gastritis.
    
    Additional medical knowledge information:
    1. 'Sharp abdominal pain' is a possible disease of 'Gastritis'.
    2. 'Gastritis' has the symptom of 'Nausea'.
    4. 'Gastritis', which causes 'Nausea', also has the symptom of 'Vomiting'.
    1. Gastritis is associated with the symptoms Sharp abdominal pain, Vomiting, Nausea.
    2. In the case of gastritis, medical tests such as Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel are required.
    3. To manage gastritis, medications like Amylases, Aluminum Hydroxide (M.A.H.), Benzocaine Topical, Famotidine, Pantoprazole, Ranitidine, Sucralfate (Carafate), Cimetidine, and Bismuth Subsalicylate (Pepto-Bismol) may be needed.
    5. In treating gastritis, patients might be prescribed medications such as Amylases, Aluminum Hydroxide (M.A.H.), Benzocaine Topical, Famotidine, Pantoprazole, Ranitidine, Sucralfate (Carafate), Cimetidine, and Bismuth Subsalicylate (Pepto-Bismol).
    
    Output:
    In order to confirm whether you have gastritis or not, I would like to recommend a few medical tests. We need to conduct Hematologic tests (Blood test), Complete blood count (Cbc), Urinalysis, Intravenous fluid replacement, Glucose measurement (Glucose level), Kidney function tests (Kidney function test), and Electrolytes panel. 
    
    **Patient Input:**
    {text}
    **Additional medical knowledge information:** 
    {pruned_kg_outputs}
    [/INST]

    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["pruned_kg_outputs", "text"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    output_all = llm_chain.run(text=text,pruned_kg_outputs=pruned_kg_outputs)
    
    return output_all

if __name__ == "__main__":
    
    graph_start_time = time.time()
    #Setting up KG in Neo4j
    print("--Connecting to Neo4j--")
    uri = "bolt://44.201.89.220:7687"
    username = "neo4j"
    password = "knife-deletion-receivers"

    driver = GraphDatabase.driver(uri, auth=(username, password), max_connection_lifetime=86400)
    session = driver.session()

    ## run this to load entire KG for the 1st time 
    # session.run("MATCH (n) DETACH DELETE n")# clean all
    # df = pd.read_csv('./data/DisTreatKG/DisTreatKG_train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])

    # for index, row in df.iterrows():
    #     head_name = row['head']
    #     tail_name = row['tail']
    #     relation_name = row['relation']

    #     query = (
    #         "MERGE (h:Entity { name: $head_name }) "
    #         "MERGE (t:Entity { name: $tail_name }) "
    #         "MERGE (h)-[r:`" + relation_name + "`]->(t)"
    #     )
    #     try:
    #         session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
    #     except:
    #         continue
    
    graph_end_time = time.time()
    graph_time_in_min = (graph_end_time - graph_start_time) / float(60)
    print("Time taken to load graph (mins): ", round(graph_time_in_min,2))
    
    ##Setting up Mixtral model
    print("--setting up Mixtral--")
    tokenizer_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    model_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    device_map = 'auto'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16,
                            device_map=device_map,
                            trust_remote_code=False,
                            revision='main')

    pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
                top_k=0,
                repetition_penalty=1.1
                )
    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})

    # ##Loading KG embeddings
    with open('./data/keyword_embeddings.pkl','rb') as f1:
        keyword_embeddings = pickle.load(f1)
    with open('./data/DisTreatKG/DisTreatKG_entity_embeddings.pkl','rb') as f2:
        entity_embeddings = pickle.load(f2)
    
    re1 = r"<CLS>(.*?)<SEP>"
    re2 = r'The extracted entities are (.*?)<END>'
    re3 = r"The extracted entity is (.*?)<END>"

    ## Setting up output file
    with open('./output/SKnowGPT_output_without_dtt_cot1_pruned_DisTreatKG_threshold_mx.csv', 'w', newline='') as f3:
        writer = csv.writer(f3)
        writer.writerow(['ID', 'Question', 'Label', 'SKnowGPT'])

    question_start_time = time.time()

    ##Loading Question from dataset
    with open("./data/GenMedGPT-5k/GenMedGPT_test_data_with_NER.json", encoding='utf-8') as f:
        data = f.read()
        data = json.loads(data)
        for i,line in enumerate(data):
            try:
                print('--',i+1,'--')
                input = line["qustion_output"]
                input = input.replace("\n","")
                input = input.replace("<OOS>","<EOS>")
                input = input.replace(":","") + "<END>"
                input_text = re.findall(re1,input)
                
                output = line["answer_output"]
                output = output.replace("\n","")
                output = output.replace("<OOS>","<EOS>")
                output = output.replace(":","") + "<END>"
                output_text = re.findall(re1,output)
                
                question_kg = re.findall(re2,input) 
                if len(question_kg) == 0:
                    question_kg = re.findall(re3,input) 
                    if len(question_kg) == 0:
                        print("<Warning> no entities found", input)
                question_kg = question_kg[0].replace("<END>","").replace("<EOS>","")
                question_kg = question_kg.replace("\n","")
                question_kg = question_kg.split(", ")
                
            except:
                continue
            
            output_all = None
            match_kg = []
            match_80 = []
            match_60 = []
            match_50 = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])   

            ##Identifying matching entities between Kg and question 
            # for kg_entity in question_kg:
                        
            #     keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            #     kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

            #     cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
            #     max_index = cos_similarities.argmax()
                            
            #     match_kg_i = entity_embeddings["entities"][max_index]
            #     while match_kg_i.replace(" ","_") in match_kg:
            #         cos_similarities[max_index] = 0
            #         max_index = cos_similarities.argmax()
            #         match_kg_i = entity_embeddings["entities"][max_index]

            #     match_kg.append(match_kg_i.replace(" ","_"))

            ##Identifying matching entities between Kg and question - Applying 3 level threshold
            for kg_entity in question_kg:
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()

                if (cos_similarities[max_index] >= 0.8):
                    match_kg_i = entity_embeddings["entities"][max_index]
                    match_kg_i = match_kg_i.replace(" ","_")
                    pattern = r"\b" + re.escape(match_kg_i) + r"\b"
                    if not any(re.search(pattern, word) for word in match_80):
                        match_80.append(match_kg_i.replace(" ","_"))
                
                elif (cos_similarities[max_index] >= 0.6):
                    match_kg_i = entity_embeddings["entities"][max_index]
                    match_kg_i = match_kg_i.replace(" ","_")
                    pattern = r"\b" + re.escape(match_kg_i) + r"\b"
                    if not any(re.search(pattern, word) for word in match_60):
                        match_60.append(match_kg_i.replace(" ","_"))
                
                elif (cos_similarities[max_index] >= 0.5):
                        match_kg_i = entity_embeddings["entities"][max_index]
                        match_kg_i = match_kg_i.replace(" ","_")
                        pattern = r"\b" + re.escape(match_kg_i) + r"\b"
                        if not any(re.search(pattern, word) for word in match_50):
                            match_50.append(match_kg_i.replace(" ","_"))
            
            if len(match_80) != 0:
                match_kg = match_80
            elif len(match_60) != 0:
                match_kg = match_60
            elif len(match_50) != 0:
                match_kg = match_50
            
            ##Find shortest path using Neo4j
            print("--Find shortest path using Neo4j--")
            if len(match_kg) != 1 or 0:
                start_entity = match_kg[0]
                candidate_entity = match_kg[1:]
                result_path_list = []
                while 1:
                    flag = 0
                    paths_list = []
                    while candidate_entity != []:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)                        
                        paths,exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity)
                        path_list = []
                        if paths == [''] or paths == []:
                            flag = 1
                            if candidate_entity == []:
                                flag = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            for p in paths:
                                path_list.append(p.split('->'))
                            if path_list != []:
                                paths_list.append(path_list)
                        
                        if exist_entity != {}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity
                    result_path = combine_lists(*paths_list)
                
                
                    if result_path != []:
                        result_path_list.extend(result_path)                
                    if flag == 1:
                        continue
                    else:
                        break
                    
                start_tmp = []
                for path_new in result_path_list:
                
                    if path_new == []:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])
                
                if len(start_tmp) == 0:
                        result_path = {}
                        single_path = {}
                else:
                    if len(start_tmp) == 1:
                        result_path = result_path_list[:5]
                    else:
                        result_path = []
                                                    
                        if len(start_tmp) >= 5:
                            for path_new in result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    result_path.append(path_new)
                                    start_tmp.remove(path_new[0])
                                if len(result_path) == 5:
                                    break
                        else:
                            count = 5 // len(start_tmp)
                            remind = 5 % len(start_tmp)
                            count_tmp = 0
                            for path_new in result_path_list:
                                if len(result_path) < 5:
                                    if path_new == []:
                                        continue
                                    if path_new[0] in start_tmp:
                                        if count_tmp < count:
                                            result_path.append(path_new)
                                            count_tmp += 1
                                        else:
                                            start_tmp.remove(path_new[0])
                                            count_tmp = 0
                                            if path_new[0] in start_tmp:
                                                result_path.append(path_new)
                                                count_tmp += 1

                                        if len(start_tmp) == 1:
                                            count = count + remind
                                else:
                                    break

                    try:
                        single_path = result_path_list[0]
                    except:
                        single_path = result_path_list
                    
            else:
                result_path = {}
                single_path = {}            
            
            ##Find neighbour entity using Neo4j
            print("--Find neighbour entity using Neo4j--")
            neighbor_list = []
            neighbor_list_disease = []
            for match_entity in match_kg:
                disease_flag = 0
                neighbors,disease = get_entity_neighbors(match_entity,disease_flag)
                neighbor_list.extend(neighbors)

                all_diseases = set()
                while disease != []:
                    all_diseases.update(disease)
                    new_disease = []
                    for disease_tmp in disease:
                        if disease_tmp in match_kg:
                            new_disease.append(disease_tmp)

                    if len(new_disease) != 0:
                        for disease_entity in new_disease:
                            disease_flag = 1
                            neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                            disease = [d for d in disease if d not in all_diseases]
                            neighbor_list_disease.extend(neighbors)
                    else:
                        for disease_entity in disease:
                            disease_flag = 1
                            neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                            disease = [d for d in disease if d not in all_diseases]
                            neighbor_list_disease.extend(neighbors)
            if len(neighbor_list)<=5:
                neighbor_list.extend(neighbor_list_disease)

            ##Verbalize KG path using LLM
            print("--Verbalize KG path using LLM--")
            if len(match_kg) != 1 or 0:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                else:
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)
                    
                    path = "\n".join(result_new_path)
                    response_of_KG_list_path = prompt_path_finding_mx(path)
                    
            else:
                response_of_KG_list_path = '{}'

            ##Verbalise neighbour entities using LLM
            print("--Verbalise neighbour entities using LLM--")
            response_of_KG_list_neighbor = []
            neighbor_new_list = []
            
            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)

            if len(neighbor_new_list) > 5:
                neighbor_input = "\n".join(neighbor_new_list[:5])
            else:
                neighbor_input = "\n".join(neighbor_new_list)
            response_of_KG_neighbor = prompt_neighbor_mx(neighbor_input)
            
            ## Pruning KG output
            print('--Pruning KG Output--')
            if len(response_of_KG_list_path) == 0:
                overall_kg_output = response_of_KG_neighbor 
            elif len(response_of_KG_neighbor) == 0:
                overall_kg_output = response_of_KG_list_path
            else:
                overall_kg_output = response_of_KG_list_path + "\n" + response_of_KG_neighbor
            pruned_kg_outputs = prune_kg_mx(overall_kg_output, input_text[0])
            
            ##Final Answer
            print("--FINAL ANSWER--")
            # output_all = final_answer_with_dtt_cot1_with_kg_mx(input_text[0],response_of_KG_list_path,response_of_KG_neighbor)
            # output_all = final_answer_with_dtt_no_cot_with_kg_mx(input_text[0],response_of_KG_list_path,response_of_KG_neighbor)
            # output_all = final_answer_without_dtt_cot1_no_kg_mx(input_text[0])
            # output_all = final_answer_without_dtt_no_cot_with_kg_mx(input_text[0],response_of_KG_list_path,response_of_KG_neighbor)
            # output_all = final_answer_without_dtt_cot1_full_kg_mx(input_text[0],response_of_KG_list_path,response_of_KG_neighbor)
            # output_all = final_answer_without_dtt_no_cot_no_kg_mx(input_text[0])
            output_all = final_answer_without_dtt_cot1_with_pruned_kg_mx(input_text[0], pruned_kg_outputs)
            # output_all = final_answer_with_dtt_cot1_with_pruned_kg_mx(input_text[0], pruned_kg_outputs)
                            
            ##writing results to file
            with open('./output/SKnowGPT_output_without_dtt_cot1_pruned_DisTreatKG_threshold_mx.csv', 'a+', newline='') as f4:
                writer = csv.writer(f4)
                writer.writerow([i+1, input_text[0], output_text[0], output_all.strip()])
                f4.flush()
    
    question_end_time = time.time()
    
    question_time_in_min = (question_end_time - question_start_time) / float(60)
    code_time_in_min = (question_end_time - graph_start_time) / float(60)
    print("Time taken for full program executed (mins): ", round(code_time_in_min,2))
    print("Time taken to load graph (mins): ", round(graph_time_in_min,2))
    print("TIme taken for full question execution (mins): ", round(question_time_in_min,2))
