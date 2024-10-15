import sys
from decouple import config
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI 
from langchain import hub
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from typing import List
import pandas as pd
import os
import pandas as pd
import time
from pprint import pprint
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
# Append the necessary paths for dataset and vector DB
sys.path.append("../")
sys.path.append("../vectorDB/")

from dataset_ingestion import Ingestor
from train_topic_model import BERTopicTrainer

class TopicCoTSelfRAG:
    def __init__(
        self,
        max_iter=5,
        nr_topics=10,
        max_doc_retrived=5,
        dataset_path="../processed_data/2wikimultihopqa/test_subsampled.jsonl",
        vectorDB_path="../vectorDB/2wikimultihopqa",
        bert_topic_model_path="../vectorDB/bertopic_model",
    ):
        # Load OpenAI API key from environment variables
        self.openai_api_key = config("OPENAI_API_KEY")
        self.gemini_api_key = config("GEMINI_API")
        self.max_iter = max_iter
        self.max_doc_retrived = max_doc_retrived

        # Initialize the Ingestor
        self.ingestor = Ingestor(dataset_path=dataset_path, openai_api_key=self.openai_api_key)
        self.vectordb = self.ingestor.load_vectordb(vectorDB_path)  # Adjust path as needed

        # Load and configure the BERTopic model trainer
        self.trainer = BERTopicTrainer(
            nr_topics=nr_topics, bert_topic_model_path=bert_topic_model_path
        )
        self.trainer.load_topic_model()

        # Initialize LLM
        # self.llm =  ChatOpenAI(model="gpt-4o",api_key=self.openai_api_key)
        self.llm=ChatGoogleGenerativeAI(
    api_key=self.gemini_api_key,  # Fetch the API key from the environment variable
    model="gemini-1.5-pro",
    convert_system_message_to_human=True
)

        # Initialize graders and chain
        self.retrieval_grader = self._create_retrieval_grader()
        self.hallucination_grader = self._create_hallucination_grader()
        self.answer_grader = self._create_answer_grader()
        self.question_rewriter = self._create_question_rewriter()
        self.rag_chain = self._create_rag_chain()

        # Create state graph
        self.workflow = StateGraph(self.GraphState)
        self.create_graph()

    class GraphState(TypedDict):
        """
        Represents the state of our graph.
        Attributes:
            question: The question to be answered
            thoughts: LLM-generated thoughts
            generation: LLM generation
            documents: List of retrieved documents
        """

        question: str
        thoughts: str
        generation: str
        documents: List[str]

    def _create_retrieval_grader(self):
        response_schemas = [
            ResponseSchema(name="score", description="a score 'yes' or 'no'", type="string")
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template="""You are a grader assessing the relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n
            Please respond in valid JSON format as follows:\n
            {format_instructions}""",
            input_variables=["question", "document"],
            partial_variables={"format_instructions": format_instructions},
        )

        return prompt | self.llm | parser

    def _create_hallucination_grader(self):
        response_schemas = [
            ResponseSchema(name="score", description="a score 'yes' or 'no'", type="string")
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in a set of facts. \n 
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation} \n
            Please respond in valid JSON format using the following instructions: \n
            {format_instructions}""",
            input_variables=["generation", "documents"],
            partial_variables={"format_instructions": format_instructions},
        )

        return prompt | self.llm | parser

    def _create_answer_grader(self):
        response_schemas = [
            ResponseSchema(name="score", description="a score 'yes' or 'no'", type="string")
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is useful to resolve a question. \n
            Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question} \n
            Please respond in valid JSON format using the following instructions: \n
            {format_instructions}""",
            input_variables=["generation", "question"],
            partial_variables={"format_instructions": format_instructions},
        )

        return prompt | self.llm | parser

    def _create_question_rewriter(self):
        prompt = PromptTemplate(
            template="""You are a question re-writer that paraphrase a question based on the provided contex to guide twoard the final answer and. \n
            Here is the initial question: \n\n {question}.
            Here is the context: \n \n {context}
            Just genrete the new question: """,
            input_variables=["question", "context"],
        )
        return prompt | self.llm | StrOutputParser()

    def _create_rag_chain(self):

        response_schemas = [
            ResponseSchema(name="answer", description="the final answer", type="string")
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()
        prompt = PromptTemplate(
            template="""You are a question answering angent that answers a question given the context. \n
            Here is the initial question: \n\n {question}. 
            Here is the context: \n\n {context}. 
            The final answer should directly be respond the question only with no extra information
            Please respond in valid JSON format using the following instructions: \n
            {format_instructions} """,
            input_variables=["question", "context"],partial_variables={"format_instructions": format_instructions},
        )
        return prompt | self.llm | parser
    def retrieve(self, state):
        # print("---RETRIEVE---")
        
        question = state["question"]

        new_topics, new_probabilities = self.trainer.get_topics_with_probabilities(question)
        assigned_topic = new_topics[0]

        metadata_filter = {"bertopic": f"Topic {assigned_topic}"}
        retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "filter": metadata_filter,
                            "k": self.max_doc_retrived},
        )
        documents = retriever.invoke(question)
        
        # print(f"len documents {len(documents)}")
        # print(documents)
  
        return {"documents": self.format_docs(documents)}

    def generate(self, state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        thoughts = state["thoughts"]
        print(thoughts)
        question = f"{question}-- {thoughts}"
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        self.last_answer = generation['answer']
        self.iter += 1
        return {"documents": documents, "generation": generation['answer']}

    def grade_documents(self, state):
        # print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        

        score = self.retrieval_grader.invoke({"question": question, "document": documents})
        if score["score"] == "yes":
            self.filtered_docs.append(documents)

        return {
            "documents": self.filtered_docs,
        }

    def transform_query(self, state):
        # print("---TRANSFORM QUERY---")
        question = state["question"]
        better_question = self.question_rewriter.invoke({"question": question, "context":state["documents"]})
        self.iter += 1
        return {"documents": state["documents"], "question": better_question}

    def decide_to_generate(self, state):
        # print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]

        # print(self.iter)
        if self.iter >= self.max_iter:
            return "generate"
        else:
            if not filtered_documents:
                return "transform_query"
            else:
                return "generate"

    def grade_generation(self, state):
        # print("---CHECK HALLUCINATIONS---")
        score = self.hallucination_grader.invoke(
            {"documents": state["documents"], "generation": state["generation"]}
        )
        if self.iter >= self.max_iter:
            return "useful"
        else:

            if score["score"] == "yes":
                score = self.answer_grader.invoke(
                    {"question": state["question"], "generation": state["generation"]}
                )
                if score["score"] == "yes":
                    return "useful"
                else:
                    return "not useful"
            else:
                return "not supported"

    def get_cot_chain(self):
        # Define the response schema to ensure the JSON format is valid
        response_schemas = [
            ResponseSchema(
                name="thoughts",
                description="The generated reasoning and chain of thought for the question.",
                type="string",
            ),
        ]

        # Initialize a structured output parser based on the response schema
        cot_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = cot_parser.get_format_instructions()

        # Define the prompt template with explicit instructions for valid JSON output
        prompt = """You are tasked with generating a chain of thought for the question: {question}. 
                    You have access to the following context: {context}. 
                    Your job is to provide a detailed chain of reasoning and final thoughts.
                    Please respond with valid JSON using the following instructions: 
                    {format_instructions}"""

        cot_prompt = PromptTemplate(
            template=prompt,
            input_variables=[ "question", "context"],
            partial_variables={"format_instructions": format_instructions},
        )

        # Return the chain of operations: prompt -> llm -> cot_parser
        return cot_prompt | self.llm | cot_parser

    def generate_cot(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        # print("---GENERATE COT---")
        question = state["question"]
        documents = state["documents"]
        cot_chain = self.get_cot_chain()
        # RAG generation
        cot = cot_chain.invoke({"question": question, "context": documents})
        # print(cot)
        return {
            "thoughts": cot["thoughts"] + state.get("thoughts", ""),
            "question": question,
        }

    def create_graph(self):
        # Initialize the workflow
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("generate_cot", self.generate_cot)  # grade documents
        # self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_node("transform_query", self.transform_query)

        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "generate_cot")
        self.workflow.add_edge("generate_cot", "generate")
        self.workflow.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        # Compile and run
        self.app = self.workflow.compile()

    def run_pipeline(self, question):
        self.iter = 0
        start_time = time.time()
        final_state = {}

        inputs = {"question": question}
        for output in self.app.stream(inputs, {"recursion_limit": 50}):
            if "generation" in output:
                self.final_answer = output["generation"]
        # Return the final generated answer
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        return final_state

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)



if __name__ == "__main__":
    dataset = "2wikimultihopqa"
    subsample = "test_subsampled"
    model = "topic_cot_self_RAG"
    top_n = 10
    max_iter = 5
    max_doc_retrived = 5
    checkpoint_path = "../results/checkpoint_{}_{}_{}.csv".format(dataset, subsample, model)
    
    # Initialize the TopicCoTSelfRAG pipeline
    pipeline = TopicCoTSelfRAG(
        vectorDB_path="../vectorDB/{}".format(dataset),
        dataset_path="../processed_data/{}/{}.jsonl".format(dataset, subsample),
        nr_topics=top_n,
        max_iter=max_iter,
        max_doc_retrived=max_doc_retrived,
    )

    # Load evaluation data
    dict_results = pipeline.ingestor.load_evaluation_data()

    # Check if there's an existing checkpoint file and load it
    if os.path.exists(checkpoint_path):
        df_checkpoint = pd.read_csv(checkpoint_path)
        start_index = len(df_checkpoint)  # Continue from the next question
        dict_results["generated_answer"] = df_checkpoint["generated_answer"].tolist()
    else:
        start_index = 0
        dict_results["generated_answer"] = []
    
    # Iterate through the evaluation dataset starting from where it left off
    for i in range(start_index, len(dict_results["question_id"])):
        question = dict_results["question_text"][i]

        try:
            # Run the pipeline for the current question
            _ = pipeline.run_pipeline(question=question)

            # Attempt to retrieve the last generated answer
            result = pipeline.last_answer if hasattr(pipeline, "last_answer") else "***"

            # Strip leading spaces and add the result to the dictionary
            result = result.lstrip() if result else ""
            dict_results["generated_answer"].append(result)

            print(f"question#{i}")
            print(
                dict_results["question_text"][i],
                "  ->  ",
                dict_results["ground_truth"][i],
                "  ->  ",
                result,
            )
        except Exception as e:
            # If there's an error during the pipeline execution, log it and append an empty string
            result = pipeline.last_answer if hasattr(pipeline, "last_answer") else "***"
            print(f"Error processing question#{i}: {e}")
            dict_results["generated_answer"].append(result)

        # Save progress after each iteration
        df_checkpoint = pd.DataFrame({
            "question_id": dict_results["question_id"][: i + 1],
            "question_text": dict_results["question_text"][: i + 1],
            "ground_truth": dict_results["ground_truth"][: i + 1],
            "generated_answer": dict_results["generated_answer"]
        })
        df_checkpoint.to_csv(checkpoint_path, index=False)

    # Save the final results to a separate file
    df_results = pd.DataFrame(dict_results)
    df_results.to_csv(
        "../results/results_gpt4o_azure_{}_{}_{}.csv".format(dataset, subsample, model), index=False
    )
