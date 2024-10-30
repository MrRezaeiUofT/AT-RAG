import os
import sys
import json
from decouple import config
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModel
from train_topic_model import BERTopicTrainer

sys.path.append("../")
from utils import get_embedding_model

# Load the OpenAI API key
openai_api_key = config("OPENAI_API_KEY")


# Ingestor Class (modified to use BERTopicTrainer)
class Ingestor:
    def __init__(
        self,
        dataset_path: str = "Nan",
        persist_directory: str = "Nan",
        openai_api_key: str = "Nan",
    ):
        self.dataset_path = dataset_path
        self.persist_directory = persist_directory
        self.openai_api_key = openai_api_key

    def load_data(self):
        """Load the dataset and extract documents and metadata."""
        with open(self.dataset_path, "r") as file:
            data = [json.loads(line) for line in file]

        documents = []
        metadatas = []

        for i, sample in enumerate(data):
            # if i > 3:  # Limiting for debugging, you can adjust as needed
            #     break
            for context in sample["contexts"]:
                documents.append(context["paragraph_text"])
                metadatas.append(
                    {
                        "idx": context["idx"],
                        "title": context.get("title", "Unknown"),  # Default to 'Unknown'
                        "is_supporting": context.get("is_supporting", "N/A"),  # Default to 'N/A'
                    }
                )

        return documents, metadatas

    def perform_topic_classification(self, documents):
        """Train or load BERTopic model for topic classification using BERTopicTrainer."""
        topics, probabilities = self.topic_model.get_topics_with_probabilities(documents)

        return topics, probabilities

    def create_vectordb(self):
        """Create the Chroma vector store with embeddings and topic metadata."""
        # Load data and metadata
        self.topic_model = BERTopicTrainer()
        self.topic_model.load_topic_model()
        documents, metadatas = self.load_data()

        # Perform BERTopic classification and obtain embeddings
        bertopic_topics, _ = self.topic_model.get_topics_with_probabilities(documents)

        # Add BERTopic topics to metadata
        for metadata, bertopic_topic in zip(metadatas, bertopic_topics):
            metadata["bertopic"] = f"Topic {bertopic_topic}"  # Add BERTopic result

        # Initialize the embedding model
        embeddings = get_embedding_model()

        # Initialize Chroma vector store without loading all texts at once
        print("Initializing Chroma vector store...")
        vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
            # You can specify other settings here if needed
        )

        # Define the maximum batch size allowed
        MAX_BATCH_SIZE = 1000

        # Process documents in batches to avoid exceeding payload limits
        total_documents = len(documents)
        print(f"Total documents to process: {total_documents}")

        for start_idx in range(0, total_documents, MAX_BATCH_SIZE):
            end_idx = min(start_idx + MAX_BATCH_SIZE, total_documents)
            batch_texts = documents[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]

            print(
                f"Processing batch {start_idx // MAX_BATCH_SIZE + 1}: "
                f"Documents {start_idx} to {end_idx - 1}"
            )

            vectordb.add_texts(texts=batch_texts, metadatas=batch_metadatas)

        # Persist the Chroma vector store for future use
        print(f"Persisting the vector store to '{self.persist_directory}'...")
        vectordb.persist()
        print("Vector store created and persisted successfully.")

        return vectordb

    def load_vectordb(self, persist_directory):
        """Load an existing vector database if available; otherwise create a new one."""
        if os.path.exists(persist_directory):
            print(f"Loading existing vector database from '{persist_directory}'...")
            embeddings = get_embedding_model()
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

    def load_evaluation_data(self):
        """
        This method loads evaluation data (question + ground-truth) and returns it as a dictionary
        """

        dict_groundtruth = {"question_id": [], "question_text": [], "ground_truth": []}

        with open(self.dataset_path, "r") as file:
            data = [json.loads(line) for line in file]

        for sample in data:
            dict_groundtruth["question_id"].append(sample["question_id"])
            dict_groundtruth["question_text"].append(sample["question_text"])
            dict_groundtruth["ground_truth"].append(sample["answers_objects"][0]["spans"][0])

        return dict_groundtruth


if __name__ == "__main__":
    dataset = "musique"
    subsample = "test_subsampled"
    top_n = 10
    # Create an instance of Ingestor with pre-trained models
    ingestor = Ingestor(
        dataset_path="../processed_data/{}/{}.jsonl".format(dataset, subsample),
        persist_directory="../vectorDB/{}".format(dataset),
        openai_api_key=openai_api_key,
    )
    vectordb = ingestor.create_vectordb()

    # Example: Ask a question and retrieve answers
    question = "Who is the father-in-law of Queen Hyojeong?"
    results = ingestor.query_question(question, top_n=top_n)