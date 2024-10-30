import os
import sys
from decouple import config
import pandas as pd
from rouge_score import rouge_scorer
import openai
import re
import json
sys.path.append('../')
sys.path.append('../vectorDB')
sys.path.append('../processed_data')

os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
openai.api_key = os.environ["OPENAI_API_KEY"]

class Evaluation:
    def __init__(self,
                 dataset,
                 subsample,
                 rag):
        self.dataset = dataset
        self.subsample = subsample
        self.rag = rag
        
        self.system_prompt = "You are an expert in evaluating the generated answers of Large LAnguage Models."
        
        self.df = pd.read_csv("../results/results_{}_{}_{}.csv".format(self.dataset,
                                                              self.subsample,
                                                              self.rag))
    
    def rouge(self):
        """
        method to calculate the ROUGE metric
        """
        
        self.df["rouge1_precision"] = 0
        self.df["rouge1_recall"] = 0
        self.df["rouge1_fmeasure"] = 0
    
        self.df["rougeL_precision"] = 0
        self.df["rougeL_recall"] = 0
        self.df["rougeL_fmeasure"] = 0
    
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
        for index, row in self.df.iterrows():
            scores = scorer.score(row["ground_truth"], 
                                  row["generated_answer"])
            self.df.at[index, "rouge1_precision"] = scores["rouge1"].precision
            self.df.at[index, "rouge1_recall"] = scores["rouge1"].recall
            self.df.at[index, "rouge1_fmeasure"] = scores["rouge1"].fmeasure
        
            self.df.at[index, "rougeL_precision"] = scores["rougeL"].precision
            self.df.at[index, "rougeL_recall"] = scores["rougeL"].recall
            self.df.at[index, "rougeL_fmeasure"] = scores["rougeL"].fmeasure
            print(scores)
    
        """ calculating the mean of the scores """ 
        dict_evaluation = {
        'ROUGE1': {
            'precision': self.df["rouge1_precision"].mean(),
            'recall': self.df["rouge1_recall"].mean(),
            'fmeasure': self.df["rouge1_fmeasure"].mean(),
            'precision_var': self.df["rouge1_precision"].var(),
            'recall_var': self.df["rouge1_recall"].var(),
            'fmeasure_var': self.df["rouge1_fmeasure"].var(),
            'precision_std': self.df["rouge1_precision"].std(),
            'recall_std': self.df["rouge1_recall"].std(),
            'fmeasure_std': self.df["rouge1_fmeasure"].std(),
        },
        'ROUGEL': {
            'precision': self.df["rougeL_precision"].mean(),
            'recall': self.df["rougeL_recall"].mean(),
            'fmeasure': self.df["rougeL_fmeasure"].mean(),
            'precision_var': self.df["rougeL_precision"].var(),
            'recall_var': self.df["rougeL_recall"].var(),
            'fmeasure_var': self.df["rougeL_fmeasure"].var(),
            'precision_std': self.df["rougeL_precision"].std(),
            'recall_std': self.df["rougeL_recall"].std(),
            'fmeasure_std': self.df["rougeL_fmeasure"].std(),
        }
        }
        dict_evaluation = pd.DataFrame(dict_evaluation)
        dict_evaluation.to_csv("ROUGE_{}_{}_{}.csv".format(dataset,
                                                           subsample,
                                                           rag),
                               index=True)
        
        return
    
    def LLMaaJ(self,
               model="gpt-4o-mini"):
        # Initialize OpenAI LLM
        client = openai.OpenAI()
        
        self.df["Correctness"] = 0
        self.df["Completeness"] = 0
        self.df["Relevance"] = 0
        self.df["Overall"] = 0
        
        for index, row in self.df.iterrows():
            question = row["question_text"]
            generated_answer = row["generated_answer"]
            ground_truth = row["ground_truth"]
            
            evaluation_prompt = f"""
            Question: {question}
            Generated Answer: {generated_answer}
            Ground Truth: {ground_truth}

            Evaluate the generated answer based on the following criteria:
            1. Correctness: How well does the generated answer match the ground truth? (0-10)
            2. Completeness: Does the answer provide all relevant information related to the question? (0-10)
            3. Relevance: Is the answer relevant to the question being asked? (0-10)


            Please provide a score from 0 to 10 for each criterion, along with an overall score, which is the average of the four criteria.
            return your answer in JSON with these keys: Correctness, Completeness, Relevance, Overall.
            """
            
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": evaluation_prompt}
                ]
                )

            llm_response = completion.choices[0].message.content
            
            json_pattern = r'\{.*?\}'
            json_match = re.search(json_pattern, llm_response, re.DOTALL)

            try:
                if json_match:
                    json_str = json_match.group(0)  # Get the matched JSON string
                    # Parse the JSON string into a Python dictionary
                    evaluation_results = json.loads(json_str)
                    self.df.at[index, "Correctness"] = evaluation_results["Correctness"]
                    self.df.at[index, "Completeness"] = evaluation_results["Completeness"]
                    self.df.at[index, "Relevance"] = evaluation_results["Relevance"]

                    self.df.at[index, "Overall"] = evaluation_results["Overall"]
                    print(evaluation_results)
                else:
                    print("The returned JSON was not valid,...passing")
                    pass
            except Exception as e:
                pass
            
        dict_evaluation = {
            "Correctness": {
                "mean": self.df["Correctness"].mean(),
                "var": self.df["Correctness"].var(),
                "std": self.df["Correctness"].std()
            },
            "Completeness": {
                "mean": self.df["Completeness"].mean(),
                "var": self.df["Completeness"].var(),
                "std": self.df["Completeness"].std()
            },
            "Relevance": {
                "mean": self.df["Relevance"].mean(),
                "var": self.df["Relevance"].var(),
                "std": self.df["Relevance"].std()
            },
            "Overall": {
                "mean": self.df["Overall"].mean(),
                "var": self.df["Overall"].var(),
                "std": self.df["Overall"].std()
            }
        }
        
        dict_evaluation = pd.DataFrame(dict_evaluation)
        dict_evaluation.to_csv("LLMaaJ_{}_{}_{}.csv".format(dataset,
                                                            subsample,
                                                            rag),
                               index=True)
        
        return
            
        
if __name__ == "__main__":
    dataset = "2wikimultihopqa"
    subsample = "test_subsampled"
    rag = "AT_RAG"
    
    evaluation = Evaluation(dataset=dataset,
                            subsample=subsample,
                            rag=rag)
    evaluation.rouge()
    evaluation.LLMaaJ(model="gpt-4o-mini")