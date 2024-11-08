# Advanced-RAG
RAG 관련 코드
```
git clone https://github.com/2024-iMOps/Advanced-RAG.git
```

## 환경
Runpod Pod Template

- runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

Runpod Pod Summary

- 4x A40 (192 GB VRAM)

- 200 GB RAM • 36 vCPU

## Install
```
cd ~/Advanced-RAG
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

## Run Upstage DP
```
cd ~/Advanced-RAG
python upstageDP.py
```
output:
- ~/Advanced-RAG/data/parsed_pdf
- ~/Advanced-RAG/data/all_documents.json

## Run RAG
```
cd ~/Advanced-RAG
python ragBaseline.py
```
output:
- ~/Advanced-RAG/data/user_data
