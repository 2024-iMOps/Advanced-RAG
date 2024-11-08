import os
import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_upstage import UpstageEmbeddings

def setup_embeddings(embed_model):
    embed_models = [
        "Upstage/solar-embedding-1-large",
        "BAAI/bge-multilingual-gemma2",
        "jinaai/jina-embeddings-v3",
        "intfloat/multilingual-e5-large-instruct",
    ]

    if embed_model in embed_models[1:]:
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        torch.cuda.empty_cache()
        
    elif embed_model == embed_models[0]:
        embeddings = UpstageEmbeddings(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            model="solar-embedding-1-large"
        )
    else:
        raise ValueError(f"지원하지 않는 embed_model입니다: {embed_model}")
    
    return embeddings
