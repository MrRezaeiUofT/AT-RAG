import sys
from decouple import config
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import hub
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from typing import List
import pandas as pd
from pprint import pprint
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
# Append the necessary paths for dataset and vector DB
sys.path.append('../')
sys.path.append('../vectorDB')

from dataset_ingestion import Ingestor
from train_topic_model import BERTopicTrainer

class DocumentProcessingPipeline:
    def __init__(self,
                 max_iter=5,
                 nr_topics=10,
                 dataset_path="../processed_data/2wikimultihopqa/test_subsampled.jsonl",
                 vectorDB_path= '../vectorDB/2wikimultihopqa',
                 bert_topic_model_path="../vectordb/bertopic_model"):
        # Load OpenAI API key from environment variables
        self.openai_api_key = config('OPENAI_API_KEY')
        self.max_iter=max_iter
        # Initialize the Ingestor
        self.ingestor = Ingestor(dataset_path=dataset_path,
                                 openai_api_key=self.openai_api_key)
        self.vectordb = self.ingestor.load_vectordb(vectorDB_path)  # Adjust path as needed

        # Load and configure the BERTopic model trainer
        self.trainer = BERTopicTrainer(nr_topics=nr_topics,
                                        bert_topic_model_path=bert_topic_model_path)
        self.trainer.load_topic_model()

        # Initialize LLM
        self.llm = OpenAI(api_key=self.openai_api_key)

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
            ResponseSchema(
                name="score",
                description="a score 'yes' or 'no'",
                type="string"
            )]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n
            use {format_instructions} for answer output""",
            input_variables=["question", "document"],
            partial_variables={"format_instructions": format_instructions}
        )
        return prompt | self.llm | parser

    def _create_hallucination_grader(self):
        response_schemas = [
            ResponseSchema(
                name="score",
                description="a score 'yes' or 'no'",
                type="string"
            )]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()
        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in a set of facts. \n 
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}
            use {format_instructions} for answer output """,
            input_variables=["generation", "documents"],
            partial_variables={"format_instructions": format_instructions}
        )
        return prompt | self.llm | parser

    def _create_answer_grader(self):
        response_schemas = [
            ResponseSchema(
                name="score",
                description="a score 'yes' or 'no'",
                type="string"
            )]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()
        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
            Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question}
            use {format_instructions} for answer output.""",
            input_variables=["generation", "question"],
            partial_variables={"format_instructions": format_instructions}
        )
        return prompt | self.llm | parser

    def _create_question_rewriter(self):
        prompt = PromptTemplate(
            template="""You are a question re-writer that improves a question for vectorstore retrieval. \n
            Here is the initial question: \n\n {question}. Improved question: """,
            input_variables=["question"],
        )
        return prompt | self.llm | StrOutputParser()

    def _create_rag_chain(self):
        prompt = hub.pull("rlm/rag-prompt")
        return prompt | self.llm | StrOutputParser()

    def retrieve(self, state):
        print("---RETRIEVE---")
        question = state["question"]
        new_topics, new_probabilities = self.trainer.get_topics_with_probabilities(question)
        assigned_topic = new_topics[0]

        metadata_filter = {"bertopic": f"Topic {assigned_topic}"}
        retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={"filter": metadata_filter})
        documents = retriever.invoke(question)
        print("dddd")
        print(documents)
        return {"documents": self.format_docs(documents), "question": question}

    def generate(self, state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        thoughts = state["thoughts"]
        question = f"question={question}-- thoughts={thoughts}"
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        self.last_answer=generation
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []

        score = self.retrieval_grader.invoke({"question": question, "document": documents})
        if score["score"] == "yes":
            filtered_docs.append(documents)

        return {"documents": filtered_docs, "question": question}

    def transform_query(self, state):
        print("---TRANSFORM QUERY---")
        question = state["question"]
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": state["documents"], "question": better_question}

    def decide_to_generate(self, state):
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        if not filtered_documents:
            return "transform_query"
        else:
            return "generate"

    def grade_generation(self, state):
        print("---CHECK HALLUCINATIONS---")
        score = self.hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]})
        self.iter+=1
        if self.iter<self.max_iter:
            if score["score"] == "yes":
                score = self.answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
                if score["score"] == "yes" :
                    return "useful"
                else:
                    return "not useful"
            else:
                return "not supported"
        else:
            return "useful"
        

    def get_cot_chain(self):

        response_schemas = [
            ResponseSchema(
                name="thoughts",
                description="the generated reasoning and chain of thought step by step for the question",
                type='string'
            ),
        ]
        # Initialize a structured output parser based on the response schema
        cot_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = cot_parser.get_format_instructions()
        prompt=""""You are a chain of thought generator for a {question} asked. 
                    Do your best to generate a short reasoning for the {question} about how it 
                    should be answered step by step: use {format_instructions} """
        cot_prompt= PromptTemplate(template=prompt,
                                    input_variables=["generation", "question"],
            partial_variables={"format_instructions": format_instructions})
        # Return the chain of operations: prompt -> llm -> category_parser
        return cot_prompt | self.llm | cot_parser
    
    def generate_cot(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE COT---")
        question = state["question"]
        cot_chain = self.get_cot_chain()
        # RAG generation
        cot = cot_chain.invoke({"question": question})
        return {
            "thoughts": cot["thoughts"],
            "question": question,
        }

    def create_graph(self):
        # Initialize the workflow
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("generate_cot", self.generate_cot)  # grade documents
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_node("transform_query", self.transform_query)

        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "generate_cot")
        self.workflow.add_edge("generate_cot", "grade_documents")
        self.workflow.add_conditional_edges("grade_documents", self.decide_to_generate, {
            "transform_query": "transform_query",
            "generate": "generate",
        })
        self.workflow.add_edge("transform_query", "retrieve")
        self.workflow.add_conditional_edges("generate", self.grade_generation, {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        })
        # Compile and run
        self.app = self.workflow.compile()
    def run_pipeline(self, question):
        self.iter=0
        
        final_state = {}
        inputs = {"question": question}
        for output in self.app.stream(inputs):
            pprint(output)
            if "generation" in output:
                final_state.update(output)
        # Return the final generated answer
        return final_state
    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    dataset = "2wikimultihopqa"
    subsample = "test_subsampled"
    model = 'topic_cot_self_RAG'
    # L = 3
    pipeline = DocumentProcessingPipeline(vectorDB_path="../vectorDB/{}".format(dataset),
                                          dataset_path="../processed_data/{}/{}.jsonl".format(dataset,
                                                                           subsample))
    
    dict_results = pipeline.ingestor.load_evaluation_data()
    dict_results = {key: value[:L] for key, value in dict_results.items()}
    dict_results["generated_answer"] = []
    # Step 2: Create an instance of SmileRAGPipeline and query

    for i in range(len(dict_results["question_id"])):
        # if i>L:
        #     break
        question = dict_results["question_text"][i]
        _= pipeline.run_pipeline(question=question)
        result = pipeline.last_answer
        result = result.lstrip()
        dict_results["generated_answer"].append(result)
        print(dict_results["question_text"][i], 
              "  ->  ", 
              dict_results["ground_truth"][i],
              "  ->  ",
              result)
    df_results = pd.DataFrame(dict_results)
    df_results.to_csv("../results/results_{}_{}_{}.csv".format(dataset,
                                                            subsample,
                                                            model),
                      index=False)