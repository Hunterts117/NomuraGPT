import os
import logging
import sys

from llama_index import (
    GPTVectorStoreIndex, 
    SimpleDirectoryReader,
    PromptHelper,
    LLMPredictor,
    ServiceContext,
    StorageContext, 
    load_index_from_storage,
    download_loader
)
from langchain.llms import OpenAI
from pathlib import Path

class NomuraGPT:
    def __init__(self, openai_api_key: str | None = None) -> None:
        # if openai_api_key is None, then it will look the enviroment variable OPENAI_API_KEY

        # rebuild storage context
        #self.storage_context = StorageContext.from_defaults(persist_dir='storage')
        # load index
        #self.index = load_index_from_storage(self.storage_context, index_id="vector_index")       

        self.chat_history = None
        self.chain = None
        self.db = None

    def ask(self, question: str) -> str:
       
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir='storage')
        # load index
        index = load_index_from_storage(storage_context, index_id="vector_index")       

        # Query index
        query_engine = index.as_query_engine()
        gpt_response = query_engine.query(question)
        
        response = self.chain({"question": question, "response": gpt_response})
        #response = gpt_response.strip()
        
        #self.chat_history.append((question, response))
        return response

    def forget(self) -> None:
        self.db = None
        self.chain = None
        self.chat_history = None
