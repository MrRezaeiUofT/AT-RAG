from langchain_community.llms import HuggingFaceTextGenInference
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from decouple import config


def get_llm():
    # Initialize and return a HuggingFaceTextGenInference model with parameters from the config
    llm = HuggingFaceTextGenInference(
        inference_server_url=config("TGI_URL"),
        max_new_tokens=config("TGI_MAX_NEW_TOKENS", cast=int),
        top_k=config("TGI_TOP_K", cast=int),
        top_p=config("TGI_TOP_P", cast=float),
        typical_p=config("TGI_TYPICAL_P", cast=float),
        temperature=config("TGI_TEMPERATURE", cast=float),
        repetition_penalty=config("TGI_REPETITION_PENALTY", cast=float),
        streaming=config("TGI_STREAMING", cast=bool),
        seed=config("TGI_SEED", cast=int),
    )
    return llm


def get_embedding_model():
    return HuggingFaceHubEmbeddings(
        model=config("TEI_URL"), huggingfacehub_api_token="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"
    )
