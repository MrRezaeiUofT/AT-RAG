import os
import json
from decouple import config
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
openai_api_key = config('OPENAI_API_KEY')
class Ingestor:
    def __init__(self, dataset_path: str='Nan', 
                 persist_directory: str='Nan', 
                 openai_api_key: str='Nan'):
        self.dataset_path = dataset_path
        self.persist_directory = persist_directory
        self.openai_api_key = openai_api_key

    def load_data(self):
        """Load the dataset and extract documents and metadata."""
        with open(self.dataset_path, 'r') as file:
            data = [json.loads(line) for line in file]

        documents = []
        metadatas = []

        for i, sample in enumerate(data):
            if i > 3:  # Limiting data for testing purposes
                break
            for context in sample['contexts']:
                documents.append(context['paragraph_text'])
                metadatas.append({
                    'idx': context['idx'],
                    'title': context.get('title', 'Unknown'),  # Default to 'Unknown'
                    'is_supporting': context.get('is_supporting', 'N/A')  # Default to 'N/A'
                })

        return documents, metadatas

    def create_vectordb(self):
        """Create the Chroma vector store with embeddings."""
        # Load data and metadata
        documents, metadatas = self.load_data()

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        # Create Chroma vector store and save it
        vectordb = Chroma.from_texts(
            texts=documents, 
            embedding=embeddings, 
            persist_directory=self.persist_directory, 
            metadatas=metadatas
        )

        # Persist the Chroma vector store for future use
        vectordb.persist()
        return vectordb

    def load_vectordb(self,persist_directory):
        """Load an existing vector database if available; otherwise create a new one."""
        if os.path.exists(persist_directory):
            print(f"Loading existing vector database from '{persist_directory}'...")
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        else:
            print(f"Vector database not found.")
            vectordb=None
            
        
        return vectordb

# Example usage:
if __name__ == "__main__":

    # Step 1: Create an instance of the Ingestor and build the vector store
    ingestor = Ingestor(
        dataset_path='../processed_data/2wikimultihopqa/test_subsampled.jsonl', 
        persist_directory='2wikimultihopqa', 
        openai_api_key=openai_api_key  # Replace with your actual API key
    )
    # vectordb = ingestor.create_vectordb()

    vectordb = ingestor.load_vectordb('2wikimultihopqa')
