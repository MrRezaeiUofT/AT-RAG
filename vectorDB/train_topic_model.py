import os
from bertopic import BERTopic
import json
import _osx_support
class BERTopicTrainer:
    def __init__(self, dataset_path: str="", bert_topic_model_path: str = 'bertopic_model', nr_topics: int = None):
        self.dataset_path = dataset_path
        self.bert_topic_model_path = bert_topic_model_path
        self.nr_topics = nr_topics  # Number of topics (optional)
        self.topic_model = None

    def load_data(self):
        """Load the dataset and extract documents."""
        with open(self.dataset_path, 'r') as file:
            data = [json.loads(line) for line in file]

        documents = [context['paragraph_text'] for sample in data for context in sample.get('contexts', [])]
        return documents

    def train_topic_model(self, documents):
        """Train BERTopic model on the provided documents."""
        print("Training BERTopic model...")

        # Option to specify the minimum topic size to control the number of topics
        self.topic_model = BERTopic(min_topic_size=50)
        topics, probabilities = self.topic_model.fit_transform(documents)

        # Optionally reduce the number of topics
        if self.nr_topics:
            print(f"Reducing topics to {self.nr_topics}...")
            self.topic_model = self.topic_model.reduce_topics(documents, nr_topics=self.nr_topics)

        # Save the trained model
        self.topic_model.save(self.bert_topic_model_path)
        print(f"BERTopic model saved to {self.bert_topic_model_path}")

        return topics, probabilities

    def load_topic_model(self):
        """Load a pre-trained BERTopic model."""
        if os.path.exists(self.bert_topic_model_path):
            print(f"Loading BERTopic model from {self.bert_topic_model_path}...")
            self.topic_model = BERTopic.load(self.bert_topic_model_path)
        else:
            raise FileNotFoundError(f"BERTopic model not found at {self.bert_topic_model_path}. Please train the model first.")
    def get_topics_with_probabilities(self, documents):
        """Get topics and probability vectors for each document."""
        if not self.topic_model:
            self.load_topic_model()

        # Transform documents to get topics and probabilities
        topics, probabilities = self.topic_model.transform(documents)

        # Return topics and corresponding probability vectors
        return topics, probabilities

    def get_topic_info(self):
        """Retrieve information about the topics."""
        if not self.topic_model:
            self.load_topic_model()

        # Get topic information (frequency, representation, etc.)
        topic_info = self.topic_model.get_topic_info()
        return topic_info
if __name__ == "__main__":
    dataset_path = '../processed_data/2wikimultihopqa/test_subsampled.jsonl'
    trainer = BERTopicTrainer(dataset_path=dataset_path, nr_topics=20)  # Reduce to 10 topics

    # Load data and train the model
    documents = trainer.load_data()
    topics, probabilities = trainer.train_topic_model(documents)
    
    # Example: Get topics and probabilities for new documents
    trainer.load_topic_model()
    new_documents = ["This is an example of a new document about AI and technology."]
    new_topics, new_probabilities = trainer.get_topics_with_probabilities(new_documents)

    # Display topics and their probabilities for the new document
    print("Topics for new documents:", new_topics)
    print("Probability vectors for new documents:", new_probabilities)

    # Get and display topic information
    topic_info = trainer.get_topic_info()
    print("Topic Information:\n", topic_info)
