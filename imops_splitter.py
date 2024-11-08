from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_table_content(text, chunk_size):
    lines = text.strip().split('\n')
    if len(lines) < 3:
        return [text]
    
    # 분할된 청크를 저장할 리스트
    chunks = []
    start_idx = 0
    current_idx = len(lines)
    
    # chunk_size를 초과하는 동안 계속 분할
    while len('\n'.join(lines[start_idx:current_idx])) > chunk_size:
        # 한 행씩 줄여가며 chunk_size 이하가 되는 지점 찾기
        current_idx -= 1
    
    # 찾은 분할 지점으로 청크 생성
    first_chunk = '\n'.join(lines[start_idx:current_idx])
    second_chunk = '\n'.join(lines[current_idx:])
    
    chunks = [first_chunk]
    if second_chunk:
        chunks.append(second_chunk)
    
    return chunks

def split_documents(documents, chunk_size, chunk_overlap):
    """
    문서를 청크 단위로 분할하는 함수
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    split_docs = []
    for doc in documents:
        # 테이블인 경우 행 단위로 분할
        if doc.metadata.get("category") == "table":
            splits = split_table_content(doc.page_content, chunk_size)
        else:
            # 일반 텍스트인 경우 기존 splitter 사용
            splits = splitter.split_text(doc.page_content)
        
        for split_content in splits:
            split_docs.append(Document(
                page_content=f"파일 이름: {doc.metadata['file_name']}\n{split_content}",
                metadata=doc.metadata
            ))
    else:
        split_docs.append(doc)
    
    return split_docs