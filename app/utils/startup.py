import torch
import faiss
from loguru import logger
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.docstore.in_memory import InMemoryDocstore 
from langchain_community.vectorstores import FAISS

from filling import ChankingDocs



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"DEVICE ACCESS: {DEVICE}")

"""
    Downloading main LLM for QnA to local dir:
"""
local_dir = '/home/retro0/cyberspace/projects/ml/llm/saulGoodman-llm/app/llm/llm_main'
repo_id = 'Qwen/Qwen3-8B'

local_model_path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir
)



"""
    Downloading Embedding model for Vector Storage:
"""
local_dir_embed = '/home/retro0/cyberspace/projects/ml/llm/saulGoodman-llm/app/llm/llm_embed'
#repo_id_embed = 'Qwen/Qwen3-Embedding-4B'
repo_id_embed = "intfloat/multilingual-e5-base"

local_embedding_model_path = snapshot_download(
    repo_id=repo_id_embed,
    local_dir=local_dir_embed
)

embeddings = HuggingFaceEmbeddings(
    model_name = local_embedding_model_path,
    model_kwargs = {'device': 'cpu'}
)

"""
    Initialize Vector Storage:
        * embedding_dim -> hidden_size from config.json
        * index -> default IndexFlatL2
        * vector storage will be empty during initialize
"""
# embedding_dim = 2560
embedding_dim = 768

index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
logger.info(f"INITED FAISS: {vector_store}")

documents = ChankingDocs()

vector_store.add_documents(documents=documents)

"""FAISS INFORMATION"""
information ={
    "total index" : vector_store.index.ntotal,
}

logger.info(information)