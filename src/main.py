from kg_utils import *
import csv
import argparse
import time
import ast

if __name__ == '__main__':

    exc_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_dataset", type=str,
                        default="MedicationQA", help="choose dataset to load.")
    parser.add_argument("--mode", type=str,
                        default="threshold", help="choose the mode: threshold or no_threshold.")
    parser.add_argument("--LLM_model", type=str,
                        default="mixtral", help="base LLM model.")
    parser.add_argument("--openai_api_keys", type=str,
                        default="", help="if the LLM_model is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature", type=float,
                        default=0.1, help="the temperature in used in LLM.")
    parser.add_argument("--output_file", type=str,
                        default="SKnowGPT_output.csv", help="Name of output file")
    parser.add_argument("--start_idx", type=int,
                        default=0, help="the start index from where to begin answering questions from the dataset.")
    parser.add_argument("--rerank_model", type=str,
                        default="medcpt", help="Reranker model.")
    parser.add_argument("--cohere_api_keys", type=str,
                        default="", help="if the rerank_model is cohere, you need add your own cohere api keys.")
    parser.add_argument("--top_n", type=int,
                        default=10, help="top n paths to keep after reranking")
    
    args = parser.parse_args()

    if args.LLM_model == "llama3":
        initialise_llama3(args.temperature, args.max_length)
    elif args.LLM_model == "mixtral":
        initialise_mixtral(args.temperature, args.max_length)
    elif args.LLM_model.startswith("gpt"):
        initialise_gpt(args.temperature, args.max_length, args.openai_api_keys, args.LLM_model)

    initialise_neo4j()
    load_kg_in_neo4j(args.read_dataset) 
    initialise_mpnet_model()

    if args.read_dataset == "MedicationQA":
        dataset = read_medqa_dataset()
    elif args.read_dataset =="GenMedGPT":
        dataset = read_genmedgpt_dataset()

    ## Setting up output file
    output_file_name = '../output/'+ args.output_file
    with open(output_file_name, 'w', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(['ID', 'Question', 'GT_Answer', 'SKnowGPT', 'Ranked_KG_output'])
        
    for qid, data in dataset.loc[args.start_idx:, ['Question', 'Answer', 'Topic_Entities']].iterrows():
        print("--", qid+1, "--")
        question = data['Question']
        # print("Question: ",question)
        answer = data['Answer']
        # print("Answer: ",answer)
        if args.read_dataset == "MedicationQA":
            keywords_dict = data['Topic_Entities']
            keys_dict = ast.literal_eval(keywords_dict)
            question_kg = list(keys_dict.values())
            question_kg = question_kg[0]
        elif args.read_dataset =="GenMedGPT":
            question_kg = data['Topic_Entities']
        # print('Q ent = ', question_kg)

        match_kg = set()
        # find similar entities - threshold
        if args.mode == "threshold":
            print("-Finding similar entities-")
            for ent in question_kg:
                found_ent = find_entity(ent)
                if found_ent:
                    similar_entities = filter_similar_entities(ent, found_ent)
                    for sim_ent in similar_entities:
                        match_kg.add(sim_ent)
        elif args.mode == "no_threshold":
            for ent in question_kg:
                found_ent = find_exact_entity(ent)
                match_kg.update(found_ent)
        match_kg = list(match_kg)
        # print("MATCH KG: ", match_kg)

        print("-Finding shortest paths-")
        paths_found = []
        if len(match_kg) != 1 or 0:
            paths_found = find_shortest_paths_between_list(match_kg)
        final_path = []
        for path in paths_found:
            triples_sentences = []
            for trip in path:
                triple = trip.replace(',',' ')
                triples_sentences.append(triple)
            result_sent = ' '.join(triples_sentences)
            final_path.append(result_sent)

        # find neighbor paths
        print("-Finding neighbor paths-")
        neighbor_list = []
        for match_entity in match_kg:
            neighbors = find_neighbors_paths(match_entity)
            neighbor_list.extend(neighbors)
        
        final_neigbor_path = []
        for neighbor in neighbor_list:
            triples_sentences = []
            for trip in neighbor:
                triple = trip.replace(',',' ')
                triples_sentences.append(triple)
            result_sent = ' '.join(triples_sentences)
            final_neigbor_path.append(result_sent)
        final_neigbor_path = list(set(final_neigbor_path))
        
        if len(final_path) == 0:
            final_kg_output = final_neigbor_path
        elif len(final_neigbor_path) == 0:
            final_kg_output = final_path
        else:
            final_kg_output = final_path + final_neigbor_path
        # print("total kg paths = ", len(final_kg_output))

        # reranking triples
        print("-Reranking-")
        if args.rerank_model == "medcpt":
            ranked_kg_output = rerank_kg_output(question, final_kg_output, args.top_n)
        elif args.rerank_model == "cohere":
            ranked_kg_output = rerank_kg_output_cohere(question, final_kg_output, args.cohere_api_keys ,args.top_n)
        # print("Ranked output: ", ranked_kg_output)
        # print("len of ranked output = ", len(ranked_kg_output))

        # Final answer
        print("-Final Answer generation-")
        if args.LLM_model == "llama3":
            output_all = final_answer_with_kg_l3(question,ranked_kg_output, args.read_dataset)
            # output_all = final_answer_only_llm_l3(question)
        elif args.LLM_model == "mixtral":
            output_all = final_answer_with_kg_mx(question,ranked_kg_output, args.read_dataset)
            # output_all = final_answer_only_llm_mx(question)
        elif args.LLM_model.startswith("gpt"):
            output_all = final_answer_with_kg_gpt(question,ranked_kg_output, args.read_dataset)
            # output_all = final_answer_only_llm_gpt(question)
        # print("SknowGPT output: ", output_all)
        
        with open(output_file_name, 'a+', newline='') as f2:
            writer = csv.writer(f2)
            writer.writerow([qid+1, question, answer, output_all, ranked_kg_output])
            # writer.writerow([qid+1, question, answer, ranked_kg_output])
            
            f2.flush()
    
    exc_end_time = time.time()
    exc_time_in_min = (exc_end_time - exc_start_time) / float(60)
    print('Program executed (in mins)', exc_time_in_min)
    