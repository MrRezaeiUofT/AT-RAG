import sys
from decouple import config
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Append necessary paths for dataset and vector DB
sys.path.append('../')
sys.path.append('../vectorDB')

# Load OpenAI API key from environment variables
openai_api_key = config('OPENAI_API_KEY')

# Import your dataset ingestion and topic model training utilities
from dataset_ingestion import Ingestor
from train_topic_model import BERTopicTrainer

# Step 1: Generate Topic for the Question
question = "Who is Ermengarde Of Tuscany's paternal grandfather?"
new_documents = [question]

def run_qa_chain(question):
    # Ingestor setup to load the vector database
    ingestor = Ingestor(openai_api_key=openai_api_key)
    vectordb = ingestor.load_vectordb('../vectorDB/2wikimultihopqa')  # Adjust path as needed

    # Load and configure the BERTopic model trainer
    trainer = BERTopicTrainer(nr_topics=10, bert_topic_model_path="../vectordb/bertopic_model")  # Reduce to 10 topics
    trainer.load_topic_model()
    new_topics, new_probabilities = trainer.get_topics_with_probabilities(new_documents)
    assigned_topic = new_topics[0]  # Assume the first topic is the most relevant

    # Step 2: Filter the VectorDB based on the assigned topic
    metadata_filter = {"bertopic": f"Topic {assigned_topic}"}
    print(metadata_filter)
    # Query the vector database with the metadata filter
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"filter": metadata_filter})

    # Retrieve documents relevant to the question
    docs = retriever.get_relevant_documents(query=question)
    # Combine all retrieved documents as context
    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
    else:
        context = "No relevant information found."
    # Step 3: Generate the answer using LLM
    qa_prompt_template = """
        {context}

        Question: {question_text}
        Answer:
    """
    qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question_text"])

    # Initialize OpenAI LLM
    llm = OpenAI(api_key=openai_api_key)
    
    # Create an LLM chain that takes the context and generates an answer
    qa_llm_chain = LLMChain(llm=llm, prompt=qa_prompt)
    result = qa_llm_chain.run({"context": context, "question_text": question})
    
    return result

# Run the QA chain
result = run_qa_chain(question)
