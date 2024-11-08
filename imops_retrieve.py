from langchain_community.vectorstores import FAISS
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain_teddynote.retrievers import EnsembleRetriever, EnsembleMethod
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
import os

def setup_retriever(split_docs, embeddings):
    # FAISS 설정
    faiss_db = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    faiss_retriever = faiss_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 15,
        }
    )

    # KiwiBM25 설정
    bm25 = KiwiBM25Retriever.from_documents(
        documents=split_docs,
        embeddings=embeddings
    )
    bm25.k = 15

    # CC 방식의 EnsembleRetriever 설정
    cc_ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25], 
        method=EnsembleMethod.CC
    )

    # Cohere Reranker 설정
    cohere_reranker = CohereRerank(
        model="rerank-multilingual-v3.0", 
        cohere_api_key=os.getenv("COHERE_API_KEY"), 
        top_n=7
    )

    # 최종 retriever 설정
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_reranker,
        base_retriever=cc_ensemble_retriever,
    )

    return compressed_retriever