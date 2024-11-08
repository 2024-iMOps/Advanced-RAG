import json
from langchain_core.documents import Document

def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    documents = [
        Document(
            page_content=data["page_content"],
            metadata=data["metadata"]
        ) for data in json_data
    ]
    
    return documents