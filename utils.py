from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredEPubLoader,
    UnstructuredWordDocumentLoader
)
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

from typing import Any
import pathlib
import logging

logging.basicConfig(encoding='utf-8', level=logging.INFO)
LOGGING = logging.getLogger()

def init_memory():
    return ConversationBufferMemory(
         memory_key='chat_history',
         return_messages=True,
         output_key='answer'
    )

MEMORY = init_memory()

class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str , **unstructured_kwargs: Any):
        super().__init__(file_path, **unstructured_kwargs, mode="elements", strategy="fast")

class DocumentLoader(object):
    supported_extension = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        'epub': UnstructuredEPubLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader
    }

class DocumentLoaderException(Exception):
    pass

def load_document(filepath: str) -> list[Document]:

    file_extention = pathlib.Path(filepath).suffix
    loader = DocumentLoader.supported_extension.get(file_extention)
    if not loader:
        raise DocumentLoaderException(
            f'invalid extension type: {file_extention}. Currently we are not supporting.'
        )
    
    loaded = loader(filepath)
    doc = loaded.load()

    #LOGGING.info(doc)

    return doc
