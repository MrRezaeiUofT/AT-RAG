import os
import json
from decouple import config
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModel
from train_tpic_model import BERTopicTrainer

# Load the OpenAI API key
openai_api_key = config('OPENAI_API_KEY')


# Ingestor Class (modified to use BERTopicTrainer)
class Ingestor:
    def __init__(self, dataset_path: str = 'Nan',
                 persist_directory: str = 'Nan',
                 openai_api_key: str = 'Nan'):
        self.dataset_path = dataset_path
        self.persist_directory = persist_directory
        self.openai_api_key = openai_api_key

        # Initialize BERTopicTrainer to manage topic model
        self.topic_model = BERTopicTrainer()
        self.topic_model.load_topic_model()

    def load_data(self):
        """Load the dataset and extract documents and metadata."""
        with open(self.dataset_path, 'r') as file:
            data = [json.loads(line) for line in file]

        documents = []
        metadatas = []

        for i, sample in enumerate(data):
            # if i > 3:  # Limiting for debugging, you can adjust as needed
            #     break
            for context in sample['contexts']:
                documents.append(context['paragraph_text'])
                metadatas.append({
                    'idx': context['idx'],
                    'title': context.get('title', 'Unknown'),  # Default to 'Unknown'
                    'is_supporting': context.get('is_supporting', 'N/A')  # Default to 'N/A'
                })

        return documents, metadatas

    def perform_topic_classification(self, documents):
        """Train or load BERTopic model for topic classification using BERTopicTrainer."""
        topics, probabilities = self.topic_model.get_topics_with_probabilities(documents)

        return topics, probabilities

    def create_vectordb(self):
        """Create the Chroma vector store with embeddings and topic metadata."""
        # Load data and metadata
        documents, metadatas = self.load_data()

        # Perform BERTopic classification and obtain embeddings
        bertopic_topics, _ = self.topic_model.get_topics_with_probabilities(documents)

        # Add BERTopic topics to metadata
        for metadata, bertopic_topic in zip(metadatas, bertopic_topics):
            metadata['bertopic'] = f"Topic {bertopic_topic}"  # Add BERTopic result

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        # Create Chroma vector store and save it
        print("Creating Chroma vector store...")
        vectordb = Chroma.from_texts(
            texts=documents,
            embedding=embeddings,
            persist_directory=self.persist_directory,
            metadatas=metadatas
        )

        # Persist the Chroma vector store for future use
        print(f"Persisting the vector store to '{self.persist_directory}'...")
        vectordb.persist()
        print("Vector store created and persisted successfully.")
        return vectordb

    def load_vectordb(self, persist_directory):
        """Load an existing vector database if available; otherwise create a new one."""
        if os.path.exists(persist_directory):
            print(f"Loading existing vector database from '{persist_directory}'...")
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        else:
            print(f"Vector database not found. Creating a new one...")
            vectordb = self.create_vectordb()

        return vectordb

    def query_question(self, question: str, top_n: int = 3):
        """Generate a topic for the question and retrieve relevant answers."""
        # Ensure vector database is loaded
        vectordb = self.load_vectordb(self.persist_directory)

        # Step 1: Convert the question to a topic using BERTopic
        question_topic, _ = self.topic_model.get_topics_with_probabilities([question])

        print(f"Question topic generated: BERTopic {question_topic}")

        # Step 2: Search the vector database based on the topic
        print(f"Searching vector database for topic {question_topic}...")

        search_results = vectordb.similarity_search(question, k=top_n)

        # Step 3: Display or return the results
        if not search_results:
            print("No results found.")
        else:
            print(f"Top {top_n} results for the question:")
            for result in search_results:
                print(result)

        return search_results


if __name__ == "__main__":
    # Create an instance of Ingestor with pre-trained models
    ingestor = Ingestor(
        dataset_path='../processed_data/2wikimultihopqa/test_subsampled.jsonl',
        persist_directory='2wikimultihopqa',
        openai_api_key=openai_api_key,
    )
    vectordb = ingestor.create_vectordb()

    # Example: Ask a question and retrieve answers
    question = "Who is the father-in-law of Queen Hyojeong?"
    results = ingestor.query_question(question, top_n=10)

    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(result['document'])
