import logging
import os
import tempfile

from langchain.chains import (
    ConversationalRetrievalChain, 
    SimpleSequentialChain, 
    OpenAIModerationChain
    )
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.base import Chain
from utils import load_document, MEMORY
#from apikey import openaiApiKey

logging.basicConfig(encoding='utf-8', level=logging.INFO)
LOGGING = logging.getLogger()

os.environ['OPENAI_API_KEY'] = '<API_KEY>'

def configure_retriever(docs: list[Document], use_compressor: bool = False) -> BaseRetriever:
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(documents=docs)
    embeddings = OpenAIEmbeddings()
    #HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    vector_db = DocArrayInMemorySearch.from_documents(splits, embedding=embeddings)
    retriever = vector_db.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 8, 'include_metadata': True})
    
    if not use_compressor:
        return retriever

    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=.2)

    return ContextualCompressionRetriever (
        base_compressor= embeddings_filter,
        base_retriever=retriever
    )  
    
def configure_chain(retriever: BaseRetriever) -> Chain:
    params = dict(
            llm =  ChatOpenAI(model='gpt-3.5-turbo', temperature=0, streaming=True),
            retriever = retriever,
            memory = MEMORY,
            max_tokens_limit = 4000
             
        )
    LOGGING.info('Calling LLM....')
    return ConversationalRetrievalChain.from_llm(**params)


def configure_retrieval_chain(files, use_compression: bool= False, use_moderation: bool = False) -> Chain :
    docs = []
    temp_directory = tempfile.TemporaryDirectory()

    for file in files:
        temp_file_path = os.path.join(temp_directory.name, file.name)
        LOGGING.info('temp_file_path', temp_file_path)
        with open(temp_file_path, 'wb') as f:
            f.write(file.getvalue())
            docs.extend(load_document(temp_file_path))

    retriever = configure_retriever(docs=docs, use_compressor=use_compression)
    chain = configure_chain(retriever=retriever)

    if not use_moderation:
        return chain
    
    #moderation_chain =  OpenAIModerationChain()

    #return SimpleSequentialChain(chains= [chain, moderation_chain])
    


