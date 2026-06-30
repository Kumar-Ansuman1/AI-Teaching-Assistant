import os
import json
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

PERSIST_DIRECTORY = "./chroma_db"


def load_json_to_langchain_docs(folder_path):

    docs = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chunks = data.get("chunks", [])

                for idx, segment in enumerate(chunks):
                    text_content = segment.get("text", "")
                    if not text_content.strip():
                        continue
                    
                    metadata = {
                        "tutorial_number": int(segment.get("tutorial_number", 0)),
                        "video_title": segment.get("video_title", ""),
                        "topic": segment.get("topic", ""),
                        "start": float(segment.get("start", 0.0)),
                        "end": float(segment.get("end", 0.0)),
                        "source": file_name
                    }

                    doc = Document(page_content=text_content, metadata=metadata)
                    docs.append(doc)

    return docs


folder_path = "./semantic_chunks"
documents = load_json_to_langchain_docs(folder_path)

print(f"Loaded {len(documents)} segments. Vectorizing ...")

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

print("Chroma Store built successfully !")