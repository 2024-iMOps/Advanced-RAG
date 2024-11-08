import re
import os
import json
import requests
from markdownify import markdownify as markdown
from dotenv import load_dotenv
load_dotenv()



api_key = os.getenv("UPSTAGE_API_KEY")
url = "https://api.upstage.ai/v1/document-ai/document-parse"
headers = {"Authorization": f"Bearer {api_key}"}



parsed_save_path = "/workspace/Advanced-RAG/data/parsed_pdf"
if not os.path.exists(parsed_save_path):
    os.makedirs(parsed_save_path)



file_path = "/workspace/Advanced-RAG/data/pdf/iM뱅크 셀프창구 서비스 설명서.pdf"
file_name = os.path.basename(file_path).split(".")[0]



files = {"document": open(file_path, "rb")}
response = requests.post(url, headers=headers, files=files)

json_data = response.json()

if "elements" not in json_data:
    print(f"API Response: {json_data}")
    raise Exception(f"Invalid API response format: {json_data}")

# HTML을 Markdown으로 변환
for element in json_data["elements"]:
    if "html" in element["content"]:
        # markdown
        html_content = element["content"]["html"]
        element["content"]["markdown"] = markdown(html_content)

with open(os.path.join(parsed_save_path, f"{file_name}.json"), "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)



docs_save_path = "/workspace/Advanced-RAG/data"



docs = []

for element in json_data["elements"]:

    page_content = element["content"]["markdown"]
    metadata = {
        "category": element["category"],
        "coordinates": element["coordinates"],
        "file_name": file_name,
        "id": element["id"],
        "page": element["page"]
    }
    
    doc = {
        "page_content": page_content,
        "metadata": metadata
    }
    docs.append(doc)

with open(os.path.join(docs_save_path, f"all_documents.json"), "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=4, ensure_ascii=False)