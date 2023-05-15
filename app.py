import streamlit as st
from streamlit_chat import message
import os
import logging
import sys
import json
import re

#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ['OPENAI_API_KEY'] = "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"

# Setup your LLM

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
from llama_index.prompts.prompts import (
    KeywordExtractPrompt,
    KnowledgeGraphPrompt,
    PandasPrompt,
    QueryKeywordExtractPrompt,
    QuestionAnswerPrompt,
    RefinePrompt,
    RefineTableContextPrompt,
    SchemaExtractPrompt,
    SimpleInputPrompt,
    SummaryPrompt,
    TableContextPrompt,
    TextToSQLPrompt,
    TreeInsertPrompt,
    TreeSelectMultiplePrompt,
    TreeSelectPrompt,
)

from langchain.llms import OpenAI
from pathlib import Path
from IPython.display import Markdown, display
from typing import List, Dict, Any
from openai.error import OpenAIError

def clear_submit():
    st.session_state["submit"] = False

def set_openai_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = os.environ['OPENAI_API_KEY']

st.markdown('<h2>NomuraGPT 🤖</h2>', unsafe_allow_html=True)
   
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'index' not in st.session_state:
    st.session_state['index'] = []


def load_index_LLM():
        # define prompt helper
        # set maximum input size
        max_input_size = 4096
        # set number of output tokens
        num_output = 1024
        # set maximum chunk overlap
        max_chunk_overlap = 20
        prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

        # define LLM
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_output))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        
        # build index
        documents = SimpleDirectoryReader('./data').load_data()
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
        # save index to disk
        index.set_index_id("vector_index")
        index.storage_context.persist('storage')
        #st.write("index built!!")
        return index

def load_index_storage():
        storage_context = StorageContext.from_defaults(persist_dir='storage')
        # load index
        index = load_index_from_storage(storage_context, index_id="vector_index")
        return index
    
def get_text():
        #st.header("Ask me:")
        input_text = st.text_area("You:", value="", on_change=clear_submit)
        return input_text

def get_answer(query_text: str):

        if not st.session_state['index']:
            index = load_index_LLM()
        else:
            index = st.session_state['index']
            #index = load_index_storage()
        
        # define custom QuestionAnswerPrompt
        #query_str = "What did the author do growing up?"
        QA_PROMPT_TMPL = (
                "We have provided context information below. \n"
                "---------------------\n"
                "{context_str}"
                "\n---------------------\n"
                "Given this information and not prior knowledge, please provide a final answer in Chinese to the question: {query_str}\n"
                "If you don't know the answer, just say that \"sorry, I don't know.\" Don't try to make up an answer. \n"
            )
        QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

        # Query your index!
        query_engine = index.as_query_engine(
            text_qa_template=QA_PROMPT
        )
        #query_engine = index.as_query_engine()

        response = query_engine.query(query_text)
        response_string = '{"outout_text":"' + str(response) + '"}' 
        res = re.sub(r'[\x00-\x1f]', '', response_string)
        answer_string = json.loads(res)
        return answer_string

    
if not st.session_state['index'] :
    st.session_state['index'] = load_index_LLM()

user_input = get_text()
button = st.button("Submit")
if button or st.session_state.get("submit"):
    if not user_input:
        st.error("Please enter a question!")
    else:
        st.session_state["submit"] = True
        try:
            if not st.session_state['index'] :
                st.session_state['index'] = load_index_LLM()
            
            answer = get_answer(user_input)
            #st.write(answer)
            #st.write(answer["outout_text"])
            
            st.session_state.past.append(user_input)
            st.session_state.generated.append(answer["outout_text"])

        except OpenAIError as e:
            st.error(e._message)
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                #st.write("i=" + str(i))
                #st.write(st.session_state["generated"][i])
                message(st.session_state["generated"][i], key=str(i))
