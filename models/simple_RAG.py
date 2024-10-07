import os
import sys
from decouple import config
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import pandas as pd
import itertools

sys.path.append("../")
sys.path.append("../vectorDB")
openai_api_key = config("OPENAI_API_KEY")
from utils import get_llm
from vectorDB.dataset_ingestion import Ingestor


class SmileRAGPipeline:
    def __init__(self, vectordb, openai_api_key):
        self.vectordb = vectordb
        self.openai_api_key = openai_api_key

        # Set up the prompt template for LLM responses
        self.prompt_template = """
        {context}

        Question: {question_text}
        Answer:
        """
        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=["context", "question_text"]
        )

        # Initialize OpenAI LLM
        self.llm = get_llm()

    def run_chain(self, question):
        """Run the RAG pipeline by querying the vector database and generating a response."""
        retriever = self.vectordb.as_retriever()
        docs = retriever.get_relevant_documents(query=question)

        # Use the first relevant document as context for simplicity, and include metadata in the output
        if docs:
            context = docs[0].page_content
            metadata = docs[0].metadata
            title = metadata.get("title", "Unknown")
            idx = metadata.get("idx", "N/A")
            is_supporting = metadata.get("is_supporting", "N/A")

            context_with_metadata = (
                f"Title: {title}\nIndex: {idx}\nSupporting: {is_supporting}\n\n{context}"
            )
        else:
            context_with_metadata = "No relevant information found."

        # Create an LLM chain that takes the context and generates an answer
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

        # Run the chain with the context and question
        result = llm_chain.run({"context": context_with_metadata, "question_text": question})

        return result


if __name__ == "__main__":
    dataset = "2wikimultihopqa"
    subsample = "test_subsampled"
    model = "simple_RAG"
    # L = 3
    ingestor = Ingestor(
        dataset_path="../processed_data/{}/{}.jsonl".format(dataset, subsample),
        openai_api_key=openai_api_key,
    )

    dict_results = ingestor.load_evaluation_data()
    # dict_results = {key: value[:L] for key, value in dict_results.items()}

    dict_results["generated_answer"] = []

    vectordb = ingestor.load_vectordb("../vectorDB/{}".format(dataset))
    # Step 2: Create an instance of SmileRAGPipeline and query
    rag_pipeline = SmileRAGPipeline(vectordb=vectordb, openai_api_key=openai_api_key)

    for i in range(len(dict_results["question_id"])):
        # if i>L:
        #     break
        question = dict_results["question_text"][i]
        result = rag_pipeline.run_chain(question)
        result = result.lstrip()
        dict_results["generated_answer"].append(result)
        print(
            dict_results["question_text"][i],
            "  ->  ",
            dict_results["ground_truth"][i],
            "  ->  ",
            result,
        )
    df_results = pd.DataFrame(dict_results)
    df_results.to_csv(
        "../results/results_{}_{}_{}.csv".format(dataset, subsample, model), index=False
    )
