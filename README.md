
# AT-RAG: An Adaptive RAG Model Enhancing Query Efficiency with Topic Filtering and Iterative Reasoning

## Overview
**AT-RAG** (Adaptive Retrieval-Augmented Generation) is a novel RAG model developed to address the challenges of complex multi-hop queries, which are often problematic for large language models (LLMs) like GPT-4. By incorporating **topic filtering** and **iterative reasoning**, AT-RAG significantly improves both retrieval efficiency and reasoning accuracy in question answering (QA).

AT-RAG leverages **BERTopic** for dynamic topic modeling, which assigns relevant topics to each incoming query, thereby boosting retrieval accuracy and computational efficiency. This model is adept at handling different QA tasks, including both general inquiries and complex, domain-specific scenarios, such as medical QA, by managing intricate multi-step queries effectively.

The figure below provides an overview of the AT-RAG model architecture, showcasing the integration of topic filtering and iterative reasoning for enhanced query efficiency and accuracy:

![AT-RAG Model Architecture](Flowchart.png)

## AT-RAG Performance
The average overall score across multiple datasets for AT-RAG demonstrates a significant performance improvement over state-of-the-art models like Adaptive RAG. The graph below shows the comparison in performance scores, along with standard deviations for each model:

![AT-RAG Model Performance](output.png)

### Paper
For more details, please refer to our paper:  
[AT-RAG: An Adaptive RAG Model Enhancing Query Efficiency with Topic Filtering and Iterative Reasoning](https://arxiv.org/abs/2410.12886)

### Key Features
- Topic Filtering: Uses BERTopic to dynamically assign relevant topics to each query, improving retrieval accuracy.
- Iterative Reasoning: Employs multistep reasoning to answer complex, multi-hop queries.
- Efficiency & Precision: Reduces retrieval time while maintaining high precision, making it suitable for both general and specialized tasks.
- Versatile Use Cases: Demonstrated effectiveness in both standard QA benchmarks and medical QA case studies.
## Datasets
You can download multi-hop datasets (MuSiQue, HotpotQA, and 2WikiMultiHopQA) from [StonyBrookNLP/ircot](https://github.com/StonyBrookNLP/ircot).

```bash
# Download the preprocessed datasets for the test set.
bash ./download/processed_data.sh
```

---

## 1. Topic Modeling with `vectorDB/train_topic_model.py`

This section describes how to perform topic modeling using the `BERTopicTrainer` class, which is based on the BERTopic algorithm. In this example, we load a dataset, train a topic model, and extract topics and their probabilities for new documents.

### Example Usage

```python
# Define dataset parameters
dataset = "2wikimultihopqa"
subsample = "test_subsampled"
dataset_path = f"../processed_data/{dataset}/{subsample}.jsonl"

# Create an instance of BERTopicTrainer and set the number of topics
trainer = BERTopicTrainer(dataset_path=dataset_path, nr_topics=20)  # Reduce to 20 topics

# Load data and train the topic model
documents = trainer.load_data()
topics, probabilities = trainer.train_topic_model(documents)

# Get topics and probabilities for new documents
trainer.load_topic_model()
new_documents = ["This is an example of a new document about AI and technology."]
new_topics, new_probabilities = trainer.get_topics_with_probabilities(new_documents)

# Display topics and their probabilities for the new document
print("Topics for new documents:", new_topics)
print("Probability vectors for new documents:", new_probabilities)

# Get and display topic information
topic_info = trainer.get_topic_info()
print("Topic Information:\n", topic_info)
```

### Parameters
- **dataset**: Name of the dataset for topic modeling (e.g., `2wikimultihopqa`).
- **subsample**: Subsample of the dataset (e.g., `test_subsampled`).
- **dataset_path**: Path to the dataset in `.jsonl` format.

### Methods
- `load_data()`: Loads the data from the specified dataset path.
- `train_topic_model(documents)`: Trains the BERTopic model using the provided documents.
- `load_topic_model()`: Loads a previously trained topic model.
- `get_topics_with_probabilities(new_documents)`: Retrieves topics and their probabilities for new documents.
- `get_topic_info()`: Retrieves information about the topics (e.g., topic labels and top words).

### Outputs
- **Topics for New Documents**: Lists topics assigned to each new document.
- **Probability Vectors**: Provides probability distribution over topics for each new document.
- **Topic Information**: Displays detailed information about each topic, including top contributing words.

---

## 2. Create the Vector DB with `vectorDB/dataset_ingestion.py`

This section demonstrates how to use the `Ingestor` class to ingest a dataset and retrieve the top `n` answers for a given question.

### Example Usage

```python
if __name__ == "__main__":
    dataset = "2wikimultihopqa"
    subsample = "test_subsampled"
    top_n = 10  # Number of top answers to retrieve

    # Create an instance of Ingestor with pre-trained models
    ingestor = Ingestor(
        dataset_path=f"../processed_data/{dataset}/{subsample}.jsonl",
        persist_directory=f"../vectorDB/{dataset}",
        openai_api_key=openai_api_key
    )

    # Create a vector database
    vectordb = ingestor.create_vectordb()

    # Example: Ask a question and retrieve top answers
    question = "Who is the father-in-law of Queen Hyojeong?"
    results = ingestor.query_question(question, top_n=top_n)

    # Display the results
    print(results)
```

### Parameters
- **dataset**: Name of the dataset for question answering (e.g., `2wikimultihopqa`).
- **subsample**: Subsample of the dataset (e.g., `test_subsampled`).
- **top_n**: Number of top answers to retrieve for the question.

### Methods
- `create_vectordb()`: Creates a vector database from the ingested dataset.

---

## 3. QA with `models/AT_RAG.py`

This section demonstrates how to use the `TopicCoTSelfRAG` class to process documents, query questions, and retrieve generated answers using a pre-trained model and vector database. The pipeline incorporates topic modeling through a CoT (Chain-of-Thought) approach and RAG (Retrieval-Augmented Generation) for answering questions.

### Example Usage

```python
if __name__ == "__main__":
    dataset = "2wikimultihopqa"
    subsample = "test_subsampled"
    model = "AT_RAG"
    top_n = 10  # Number of topics to retrieve
    max_iter = 5  # Maximum iterations for topic modeling

    # Create an instance of TopicCoTSelfRAG
    pipeline = TopicCoTSelfRAG(
        vectorDB_path=f"../vectorDB/{dataset}",
        dataset_path=f"../processed_data/{dataset}/{subsample}.jsonl",
        nr_topics=top_n,
        max_iter=max_iter
    )

    # Load evaluation data
    dict_results = pipeline.ingestor.load_evaluation_data()
    dict_results["generated_answer"] = []

    # Generate answers for each question
    for i in range(len(dict_results["question_id"])):
        question = dict_results["question_text"][i]
        _ = pipeline.run_pipeline(question=question)
        result = pipeline.last_answer.strip()
        dict_results["generated_answer"].append(result)

        print(f"{question} -> {dict_results['ground_truth'][i]} -> {result}")

    # Save results to CSV
    pd.DataFrame(dict_results).to_csv(
        f"../results/results_{dataset}_{subsample}_{model}.csv", index=False
    )
```

### Parameters
- **dataset**: Dataset for document processing and question answering (e.g., `2wikimultihopqa`).
- **subsample**: Subsample of the dataset (e.g., `test_subsampled`).
- **model**: Model used for generating answers (e.g., `AT_RAG`).
- **top_n**: Number of topics to retrieve during topic modeling.
- **max_iter**: Maximum iterations for the answering pipline.

### Methods
- `load_evaluation_data()`: Loads evaluation data containing questions and ground truth.
- `run_pipeline(question)`: Runs the pipeline for a given question, including topic modeling and answer generation.
- `last_answer`: Retrieves the last generated answer.

---

## 4. Dataset Evaluation with `evaluation/evaluate.py`

This section demonstrates how to use the `Evaluation` class to evaluate the results of a dataset using ROUGE scores and the LLMaaJ (Large Language Model as a Judge) framework with GPT models.

### Example Usage

```python
if __name__ == "__main__":
    dataset = "2wikimultihopqa"
    subsample = "test_subsampled"
    rag = "AT_RAG"

    # Create an instance of Evaluation
    evaluation = Evaluation(dataset=dataset, subsample=subsample, rag=rag)

    # Perform ROUGE evaluation
    evaluation.rouge()

    # Perform LLM-based evaluation using GPT-4o-mini
    evaluation.LLMaaJ(model="gpt-4o-mini")
```

### Parameters
- **dataset**: Name of the dataset used for evaluation (e.g., `2wikimultihopqa`).
- **subsample**: Subsample of the dataset (e.g., `test_subsampled`).
- **rag**: RAG model used (e.g., `AT_RAG`).
- **model**: Model for LLM-based evaluation (e.g., `gpt-4o-mini`).

### Methods
- `rouge()`: Evaluates the dataset using ROUGE metrics.
- `LLMaaJ(model)`: Uses an LLM to evaluate the quality of generated answers.
## Final Results
- Final results are saved in the results directory
---

## Citation
When utilizing this repository to generate published results or integrate it into other software, kindly acknowledge our paper by citing it.
```bibtex
@article{rezaei2024rag,
  title={AT-RAG: An Adaptive RAG Model Enhancing Query Efficiency with Topic Filtering and Iterative Reasoning},
  author={Rezaei, Mohammad Reza and Hafezi, Maziar and Satpathy, Amit and Hodge, Lovell and Pourjafari, Ebrahim},
  journal={arXiv preprint arXiv:2410.12886},
  year={2024}
}
```
