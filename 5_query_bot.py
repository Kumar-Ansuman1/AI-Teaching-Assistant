from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

PERSIST_DIRECTORY = "./chroma_db"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5}
)

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)


def search_tutorials(user_query):

    print(f"\nSearching database for: '{user_query}'")
    print("=" * 60)

    results = compression_retriever.invoke(user_query)

    if not results:
        print("No matching segments found.")
        return
    
    for i, doc in enumerate(results):
        meta = doc.metadata
        print(f"\n[Match #{i+1}]")
        print(f" Video Title : {meta.get('video_title')}")
        print(f" Tutorial No. : {meta.get('tutorial_number')}")
        print(f" Timestamps   : {meta.get('start')}s to {meta.get('end')}s")
        print(f" Transcript   : \"{doc.page_content}\"")
        print("-" * 40)


if __name__ == "__main__":
    print("============================================================")
    print("Welcome to the Tutorial Timestamp Search Tool!")
    print("Type 'exit' or 'quit' at any time to close the program.")
    print("============================================================")

    while True:
        user_input = input("\n🔍 Enter your search question: ")
        cleaned_input = user_input.strip()
        if cleaned_input.lower() in ['exit', 'quit', 'q']:
            print("\nExiting search tool. Happy coding!")
            break
        if not cleaned_input:
            print("⚠️ Please enter a valid question.")
            continue
        search_tutorials(cleaned_input)