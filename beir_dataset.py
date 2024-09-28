import os
import pathlib
import logging
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

class BEIRRetriever:
    def __init__(self, dataset_name, model_name="msmarco-distilbert-base-tas-b", batch_size=16, score_function="dot"):
        """
        Initialize the BEIRRetriever class with dataset name, model, batch size, and score function.
        :param dataset_name: Name of the dataset to be downloaded and loaded (e.g., "scifact").
        :param model_name: Name of the SentenceBERT model to be used for retrieval (default: msmarco-distilbert-base-tas-b).
        :param batch_size: Batch size for model inference (default: 16).
        :param score_function: Scoring function for the retrieval (default: "dot"). Use "cos_sim" for cosine similarity.
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.score_function = score_function
        self.model = None
        self.retriever = None
        self.data_path = None
        self.corpus = None
        self.queries = None
        self.qrels = None

        # Set up logging
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO,
                            handlers=[LoggingHandler()])

    def download_and_prepare_data(self):
        """
        Download and prepare the dataset for retrieval.
        """
        logging.info(f"Downloading and preparing dataset: {self.dataset_name}")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        self.data_path = util.download_and_unzip(url, out_dir)

    def load_data(self):
        """
        Load the corpus, queries, and qrels from the prepared dataset.
        """
        logging.info(f"Loading data for dataset: {self.dataset_name}")
        self.corpus, self.queries, self.qrels = GenericDataLoader(data_folder=self.data_path).load(split="test")

    def load_model(self):
        """
        Load the SBERT model for dense retrieval.
        """
        logging.info(f"Loading model: {self.model_name}")
        self.model = DRES(models.SentenceBERT(self.model_name), batch_size=self.batch_size)
        self.retriever = EvaluateRetrieval(self.model, score_function=self.score_function)

    def retrieve(self):
        """
        Perform retrieval on the loaded data.
        """
        logging.info(f"Retrieving results for dataset: {self.dataset_name}")
        results = self.retriever.retrieve(self.corpus, self.queries)
        return results

    def evaluate(self, results):
        """
        Evaluate the model using NDCG@k, MAP@K, Recall@K, and Precision@K.
        :param results: Retrieved results to evaluate.
        """
        logging.info(f"Evaluating results for dataset: {self.dataset_name}")
        ndcg, _map, recall, precision = self.retriever.evaluate(self.qrels, results, self.retriever.k_values)
        return ndcg, _map, recall, precision

# Example usage
if __name__ == "__main__":
    dataset_name = "scifact"  # Replace with any BEIR dataset
    retriever = BEIRRetriever(dataset_name)
    
    retriever.download_and_prepare_data()
    retriever.load_data()
    retriever.load_model()
    
    results = retriever.retrieve()
    
    ndcg, _map, recall, precision = retriever.evaluate(results)
    
    print(f"NDCG: {ndcg}")
    print(f"MAP: {_map}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")

