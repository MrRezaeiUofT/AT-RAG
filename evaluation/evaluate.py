import os
import sys
from decouple import config
import pandas as pd
from rouge_score import rouge_scorer
import openai
import re
import json
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate

sys.path.append("../")
sys.path.append("../vectorDB")
sys.path.append("../processed_data")
from utils import get_llm

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
openai.api_key = os.environ["OPENAI_API_KEY"]


class Evaluation:
    def __init__(self, dataset, subsample, rag):
        self.dataset = dataset
        self.subsample = subsample
        self.rag = rag

        self.system_prompt = (
            "You are an expert in evaluating the generated answers of Large LAnguage Models."
        )

        self.df = pd.read_csv(
            "../results/results_{}_{}_{}.csv".format(self.dataset, self.subsample, self.rag)
        )

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

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

        for index, row in self.df.iterrows():
            scores = scorer.score(row["ground_truth"], row["generated_answer"])
            self.df.at[index, "rouge1_precision"] = scores["rouge1"].precision
            self.df.at[index, "rouge1_recall"] = scores["rouge1"].recall
            self.df.at[index, "rouge1_fmeasure"] = scores["rouge1"].fmeasure

            self.df.at[index, "rougeL_precision"] = scores["rougeL"].precision
            self.df.at[index, "rougeL_recall"] = scores["rougeL"].recall
            self.df.at[index, "rougeL_fmeasure"] = scores["rougeL"].fmeasure
            print(scores)

        """ calculating the mean of the scores """
        dict_evaluation = {
            "ROUGE1": {
                "precision": self.df["rouge1_precision"].mean(),
                "recall": self.df["rouge1_recall"].mean(),
                "fmeasure": self.df["rouge1_fmeasure"].mean(),
                "precision_var": self.df["rouge1_precision"].var(),
                "recall_var": self.df["rouge1_recall"].var(),
                "fmeasure_var": self.df["rouge1_fmeasure"].var(),
                "precision_std": self.df["rouge1_precision"].std(),
                "recall_std": self.df["rouge1_recall"].std(),
                "fmeasure_std": self.df["rouge1_fmeasure"].std(),
            },
            "ROUGEL": {
                "precision": self.df["rougeL_precision"].mean(),
                "recall": self.df["rougeL_recall"].mean(),
                "fmeasure": self.df["rougeL_fmeasure"].mean(),
                "precision_var": self.df["rougeL_precision"].var(),
                "recall_var": self.df["rougeL_recall"].var(),
                "fmeasure_var": self.df["rougeL_fmeasure"].var(),
                "precision_std": self.df["rougeL_precision"].std(),
                "recall_std": self.df["rougeL_recall"].std(),
                "fmeasure_std": self.df["rougeL_fmeasure"].std(),
            },
        }
        dict_evaluation = pd.DataFrame(dict_evaluation)
        dict_evaluation.to_csv("ROUGE_{}_{}_{}.csv".format(dataset, subsample, rag), index=True)

        return

    def LLMaaJ(self, model):
        # Initialize OpenAI LLM
        client = model

        # Define response schemas for each evaluation criterion
        response_schemas = [
            ResponseSchema(
                name="Correctness", description="Score for Correctness (0-10)", type="integer"
            ),
            ResponseSchema(
                name="Completeness", description="Score for Completeness (0-10)", type="integer"
            ),
            ResponseSchema(
                name="Relevance", description="Score for Relevance (0-10)", type="integer"
            ),
            ResponseSchema(name="Clarity", description="Score for Clarity (0-10)", type="integer"),
            ResponseSchema(
                name="Overall",
                description="Overall score, average of other four (0-10)",
                type="integer",
            ),
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        # Create a prompt template using the new approach
        prompt = PromptTemplate(
            template="""You are an evaluator. Please assess the generated answer based on the following question, generated answer, and ground truth:
            \n ------- \n
            Question: {question}
            \n Generated Answer: {generation}
            \n Ground Truth: {ground_truth}
            \n ------- \n
            Please provide the scores for the following criteria using {format_instructions}.
            """,
            input_variables=["question", "generation", "ground_truth"],
            partial_variables={"format_instructions": format_instructions},
        )

        self.df["Correctness"] = 0
        self.df["Completeness"] = 0
        self.df["Relevance"] = 0
        self.df["Clarity"] = 0
        self.df["Overall"] = 0
        response_chain = prompt | client | parser
        for index, row in self.df.iterrows():
            question = row["question_text"]
            generated_answer = row["generated_answer"]
            ground_truth = row["ground_truth"]

            # Call the LLM with the simple prompt

            try:
                parsed_response = response_chain.invoke(
                    {
                        "question": question,
                        "generation": generated_answer,
                        "ground_truth": ground_truth,
                    }
                )
                print(parsed_response)
                self.df.at[index, "Correctness"] = parsed_response["Correctness"]
                self.df.at[index, "Completeness"] = parsed_response["Completeness"]
                self.df.at[index, "Relevance"] = parsed_response["Relevance"]
                self.df.at[index, "Clarity"] = parsed_response["Clarity"]
                self.df.at[index, "Overall"] = parsed_response["Overall"]
            except Exception as e:
                print(f"Parsing error: {e}")
                pass

        # Calculate statistics
        dict_evaluation = {
            "Correctness": {
                "mean": self.df["Correctness"].mean(),
                "var": self.df["Correctness"].var(),
                "std": self.df["Correctness"].std(),
            },
            "Completeness": {
                "mean": self.df["Completeness"].mean(),
                "var": self.df["Completeness"].var(),
                "std": self.df["Completeness"].std(),
            },
            "Relevance": {
                "mean": self.df["Relevance"].mean(),
                "var": self.df["Relevance"].var(),
                "std": self.df["Relevance"].std(),
            },
            "Clarity": {
                "mean": self.df["Clarity"].mean(),
                "var": self.df["Clarity"].var(),
                "std": self.df["Clarity"].std(),
            },
            "Overall": {
                "mean": self.df["Overall"].mean(),
                "var": self.df["Overall"].var(),
                "std": self.df["Overall"].std(),
            },
        }

        # Convert the evaluation results to a DataFrame and save as CSV
        dict_evaluation_df = pd.DataFrame(dict_evaluation)
        dict_evaluation_df.to_csv(f"LLMaaJ_{dataset}_{subsample}_{rag}.csv", index=True)

        return


if __name__ == "__main__":
    dataset = "2wikimultihopqa"
    subsample = "test_subsampled"
    rag = "simple_RAG"

    evaluation = Evaluation(dataset=dataset, subsample=subsample, rag=rag)
    evaluation.rouge()
    evaluation.LLMaaJ(model=get_llm())
