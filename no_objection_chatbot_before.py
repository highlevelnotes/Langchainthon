import os
import uuid
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

os.environ['OPENAI_API_KEY'] = 'your_api_key'

@st.cache_resource
def load_and_split_pdf(file_path):
  loader = PyPDFLoader(file_path)
  return loader.load_and_split()

@st.cache_resource
def get_vector_store():
  persist_directory = './vector_db/large_recursive_500_0' 
  if os.path.exists(persist_directory):
    return Chroma(
      persist_directory=persist_directory,
      embedding_function=OpenAIEmbeddings(model='text-embedding-3-large')
    )
  else:
    return
  
@st.cache_resource
def initialize_components(selected_model):
  vectorstore = get_vector_store()
  retriever = vectorstore.as_retriever()

  contextualize_q_system_prompt = '''Given a chat history and the latest user question \
  which might reference context in the chat history, formulate a standalone question \
  which can be understood without the chat history. Do NOT answer the question, \
  just reformulate it if needed and otherwise return it as is.'''
  contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
      ('system', contextualize_q_system_prompt),
      MessagesPlaceholder('chat_history'),
      ('human', '{input}')
    ]
  )

  qa_system_prompt = """You are an assistant for question-answering tasks. \
  Use the following pieces of retrieved context to answer the question. \
  If you don't know the answer, just say that you don't know. \
  Keep the answer perfect. please use imogi with the answer.\
  대답은 한국어로 하고, 존댓말을 써줘. \
  
  {context}"""
  qa_prompt = ChatPromptTemplate.from_messages(
    [
      ('system', qa_system_prompt),
      MessagesPlaceholder('chat_history'),
      ('human', '{input}')
    ]
  )

  llm = ChatOpenAI(model=selected_model, streaming=True)
  history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
  question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
  rag_chain = create = create_retrieval_chain(history_aware_retriever, question_answer_chain)
  return rag_chain

st.title('이의없음!')
if 'messages' not in st.session_state:
  st.session_state['messages'] = [{'role': 'assistant', 'content': '저작권에 대해 궁금하신가요? 무엇이든 물어보세요!'}]

chat_history = StreamlitChatMessageHistory(key='chat_messages')

for msg in chat_history.messages:
  st.chat_message(msg.type).write(msg.content)

rag_chain = initialize_components('gpt-4o-mini')

conversational_rag_chain = RunnableWithMessageHistory(
  rag_chain,
  lambda session_id: chat_history,
  input_messages_key='input',
  history_messages_key='chat_history',
  output_messages_key='answer'
)

if prompt_message := st.chat_input('Your question'):
  st.chat_message('human').write(prompt_message)
  with st.chat_message('ai'):
    with st.spinner('Thinking...'):
      config = {'configurable':{'session_id':'any'}}
      response = conversational_rag_chain.invoke(
        {'input': prompt_message},
        config,
      )

      answer = response['answer']
      st.write(answer)
      with st.expander('참고 문서 확인'):
        for doc in response['context']:
          st.markdown(doc.metadata['source'], help=doc.page_content)
