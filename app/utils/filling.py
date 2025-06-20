from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

def ChankingDocs():
    files_path = [
        "/home/retro0/cyberspace/projects/ml/llm/saulGoodman-llm/app/data/_Гражданский_кодекс_Российской_Федерации_часть_вторая.pdf",
        "/home/retro0/cyberspace/projects/ml/llm/saulGoodman-llm/app/data/_Гражданский_кодекс_Российской_Федерации_часть_первая.pdf",
        "/home/retro0/cyberspace/projects/ml/llm/saulGoodman-llm/app/data/_Гражданский_кодекс_Российской_Федерации_часть_третья.pdf",
        "/home/retro0/cyberspace/projects/ml/llm/saulGoodman-llm/app/data/_Гражданский_кодекс_Российской_Федерации_часть_четверт.pdf"
    ]

    documents = []

    for path in files_path:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
        #logger.info(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 4096,
        chunk_overlap = 128
    )

    chunk_docs = text_splitter.split_documents(documents)
    
    return chunk_docs