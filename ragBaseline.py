import os
import json
import sys
import openai

from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

from imops_prompts import (
    LLMPrompt, 
    IntentPrompt, 
    IntentNOPrompt, 
    EvaluationPrompt
)
from imops_document_loader import load_documents
from imops_splitter import split_documents
from imops_embedding import setup_embeddings
from imops_llm import setup_llm
from imops_chart import evals_chart
from imops_bbox import draw_coordinates
from imops_retrieve import setup_retriever

import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

# seed
import torch
import random
from transformers import set_seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
set_seed(seed)


# load dummy data
with open('./dummydata.json', 'r') as f:
    dummydata = json.load(f)


# format docs
def format_docs(docs):
    global references
    references = docs
    context = ""
    for doc in docs:
        context += "\n\n" + doc.page_content
    return context


# LLM Prompt (intent yes prompt)
template = LLMPrompt()
llm_prompt = PromptTemplate.from_template(template)

# Evaluation Prompt
evaluation_prompt = EvaluationPrompt()

# intent prompt
template = IntentPrompt()
intent_prompt = PromptTemplate.from_template(template)

# intent no prompt
template = IntentNOPrompt()
intent_no_prompt = PromptTemplate.from_template(template)


# save directory
id = dummydata[0]['id']
exp = dummydata[0]['exp_id']

base_path = "/workspace/Advanced-RAG/data"

save_dir = f"{base_path}/user_data/id_{id}/id_{exp}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# load documents
documents = load_documents(f"{base_path}/all_documents.json")

# Text Splitter
chunk_size = dummydata[0]['chunk_size']
chunk_overlap = dummydata[0]['chunk_overlap']
split_docs = split_documents(documents, chunk_size, chunk_overlap)

# Embedding
embed_model = dummydata[0]['embed_model']
embeddings = setup_embeddings(embed_model)

# LLM
llm_model = dummydata[0]['llm_model']
llm_temperature = dummydata[0]['llm_temperature']
llm_max_tokens = dummydata[0]['llm_max_tokens']
llm = setup_llm(llm_model, llm_temperature, llm_max_tokens)

# Retriever
retriever = setup_retriever(split_docs, embeddings)


# ===============================================================================================

# Inference
question = dummydata[0]['question']

## User Intent
llm_intent = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.1,
    max_tokens=512,
)

intent_chain = (
    {"question": RunnablePassthrough()}
    | intent_prompt
    | llm_intent
    | StrOutputParser()
)
intent_answer = intent_chain.invoke(question).strip()

## User Intent: no
if "no" in intent_answer:
    intent_no_chain = (
        {"question": RunnablePassthrough()}
        | intent_no_prompt
        | llm
        | StrOutputParser()
    )
    answer = intent_no_chain.invoke(question).strip()

    print(f"Answer: {answer}")

    save_no_answer = {
        "question": question,
        "answer": answer
    }

    with open(f'{save_dir}/no_answer.json', 'w', encoding='utf-8') as f:
        json.dump(save_no_answer, f, ensure_ascii=False, indent=4)

    sys.exit()

## User Intent: yes
elif "yes" in intent_answer:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )

    evaluation_results = []

    for iteration in range(5):
        print(f"\n=== Iteration {iteration + 1} ===")
        
        print(f"Question: {question}")
        answer = rag_chain.invoke(question).strip()
        print(f"Answer: {answer}\n")

        ### 답변 평가 수행
        eval_prompt = evaluation_prompt + f"\n질문: {question}\n답변: {answer}"
        
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': '당신은 RAG 결과를 평가하는 평가자입니다.'},
                {'role': 'user', 'content': eval_prompt}],
            temperature=0
        )

        evaluation = response.choices[0].message.content
        evaluation_json = json.loads(evaluation)
        
        evaluation_results.append({
            "iteration": iteration,
            "question": question,
            "answer": answer,
            "evaluation": {
                "accuracy": evaluation_json['accuracy'],
                "conciseness": evaluation_json['conciseness'],
                "feedback": evaluation_json['feedback']
            }
        })

        print(f"이해도: {evaluation_json['accuracy']}/10점")
        print(f"간결성: {evaluation_json['conciseness']}/10점\n")
        print(f"피드백: {evaluation_json['feedback']}\n")


# 참고 문서는 한 번만 저장 (질문이 같기 때문에)
ref_page_content = [reference.page_content for reference in references]
ref_mata_data = [reference.metadata for reference in references]

save_ref = []
for i in range(len(references)):
    save_data = {
        "page_content": ref_page_content[i],
        "metadata": {
            "coordinates": ref_mata_data[i]["coordinates"],
            "file_name": ref_mata_data[i]["file_name"],
            "id": ref_mata_data[i]["id"],
            "page": ref_mata_data[i]["page"],
            "relevance_score": ref_mata_data[i]["relevance_score"]
        }
    }
    save_ref.append(save_data)

with open(f'{save_dir}/references.json', 'w', encoding='utf-8') as f:
    json.dump(save_ref, f, ensure_ascii=False, indent=4)


# 답변 평가 저장 (5개 리스트)
with open(f'{save_dir}/evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
evals_chart(evaluation_results, save_dir) # 5개 답변 평가 중 정량적인 지표 시각화


# 참조 지식 페이지 BBox
json_path = f"{base_path}/user_data/id_{id}/id_{exp}/references.json"
pdf_base_path = f"{base_path}/pdf"
draw_coordinates(json_path, pdf_base_path, save_dir)
