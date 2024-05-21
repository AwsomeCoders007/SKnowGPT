import pandas as pd
from langchain import HuggingFacePipeline, PromptTemplate,  LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import csv
import json
import time
import torch

def final_baseline_mx(text):
    template = """<s>[INST]
    You are an excellent AI doctor, and you can provide medical advice to patient's query.\
    If you don't know the answer to a question, refrain from sharing false information. \
    Do not ask any follow-up questions. Provide consise answers. Think step by step.
    [/INST]
    **Patient Input:**
    {text}
    """
    
    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    output_all = llm_chain.invoke(text)
    
    return output_all['text']

if __name__ == "__main__":
    
    code_start_time = time.time()
    ##Setting up Model
    print("--setting up Model--")
    tokenizer = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    device_map = 'auto'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
                            model_name,
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

    ## Setting up output file
    with open('./output/baseline_mx.csv', 'w', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(['ID', 'Question', 'Label', 'Mixtral'])

    question_start_time = time.time()
    re1 = r"<CLS>(.*?)<SEP>"
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
            except:
                continue
            
            ##Final Answer
            print("--FINAL ANSWER--")
            output_all = final_baseline_mx(input_text[0])
                
            ##writing results to file
            with open('./output/baseline_mx.csv', 'a+', newline='') as f2:
                writer = csv.writer(f2)
                writer.writerow([i+1, input_text[0], output_text[0], output_all.strip()])
                f2.flush()
    
    question_end_time = time.time()
    
    question_time_in_min = (question_end_time - question_start_time) / float(60)
    code_time_in_min = (question_end_time - code_start_time) / float(60)
    print("Time taken for full program executed (mins): ", round(code_time_in_min,2))
    print("TIme taken for full question execution (mins): ", round(question_time_in_min,2))