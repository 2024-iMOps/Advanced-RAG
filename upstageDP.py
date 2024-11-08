import os
import json
import requests
from markdownify import markdownify as markdown
from dotenv import load_dotenv
load_dotenv()


class PDFParser:
    def __init__(self):
        """
        Initialize
        """
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        self.url = "https://api.upstage.ai/v1/document-ai/document-parse"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

        self.base_path = "/workspace/Advanced-RAG/data"
        self.parsed_save_path = os.path.join(self.base_path, "parsed_pdf")
        if not os.path.exists(self.parsed_save_path):
            os.makedirs(self.parsed_save_path)

    def parsing(self, file_path):
        """
        Document Parse with Upstage DP
        HTML to Markdown
        """
        self.file_name = os.path.basename(file_path).split(".")[0]

        files = {"document": open(file_path, "rb")}
        response = requests.post(self.url, headers=self.headers, files=files)

        json_data = response.json()

        if "elements" not in json_data:
            print(f"API Response: {json_data}")
            raise Exception(f"Invalid API response format: {json_data}")

        # HTML을 Markdown으로 변환
        for element in json_data["elements"]:
            if "html" in element["content"]:
                html_content = element["content"]["html"]
                element["content"]["markdown"] = markdown(html_content)

        # Save
        self._save_parsed_data(json_data)

        return json_data

    def create_docs(self, json_data):
        """
        page_content, metadata
        """
        # page_content, metadata
        docs = []

        for element in json_data["elements"]:

            page_content = element["content"]["markdown"]
            metadata = {
                "category": element["category"],
                "coordinates": element["coordinates"],
                "file_name": self.file_name,
                "id": element["id"],
                "page": element["page"]
            }
            
            doc = {
                "page_content": page_content,
                "metadata": metadata
            }
            docs.append(doc)

        # Save
        self._save_docs(docs)

        return docs

    def _save_parsed_data(self, json_data):
        """
        Save Parsed Data
        """
        save_path = os.path.join(self.parsed_save_path, f"{self.file_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

    def _save_docs(self, docs):
        """
        Save Docs
        기존 데이터 유지하면서 추가
        """
        save_path = os.path.join(self.base_path, f"all_documents.json")
        
        # 기존 데이터 불러오기
        existing_docs = []
        if os.path.exists(save_path): 
            with open(save_path, "r", encoding="utf-8") as f:
                existing_docs = json.load(f)

        # 새로운 데이터 추가
        existing_docs.extend(docs)

        # Save        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(existing_docs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    pdf_dir = "/workspace/Advanced-RAG/data/pdf"
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    parser = PDFParser()
    for pdf_file in pdf_files:
        pdf_name = os.path.basename(pdf_file).split('.')[0]
        try:
            print(f"Parsing... {pdf_name}")

            json_data = parser.parsing(pdf_file)
            docs = parser.create_docs(json_data)

            print(f"Success!\nlength: {len(docs)}\n")
        except Exception as e:
            print(f"Error '{pdf_name}': {str(e)}\n")
            continue