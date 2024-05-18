#!/usr/bin/env python
# coding: utf-8

# * Hacer scrapin con beutifoulsoup en dia y en mercadona selenium por que tiene credenciales
# * Cargar los datos de los "documentos largos".
# * Dividir los documentos en secciones cortas llamadas fragmentos (chunks).
# * Transformar los fragmentos en vectores númericos (Embeddings)
# * Guardar los embeddings en una base de datos vectorial (Pinecone)
# * Realizar las consultas

# In[1]:


pip install requests


# In[2]:


import os
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


# In[3]:


load_dotenv(find_dotenv(), override=True)


# In[4]:


os.environ["PINECONE_API_KEY"] = "9aa9eec2-6465-43d6-a740-529299b29f77"


# ### Cargar Multiples URL (SCRAPING DIA)

# In[5]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  nest_asyncio')

import nest_asyncio

nest_asyncio.apply()


# In[6]:


def desde_web2(web):
    from langchain_community.document_loaders import WebBaseLoader

    # Create an instance of WebBaseLoader with the list of URLs
    loader = WebBaseLoader(web)

    # Set the number of requests per second to control the scraping rate
    loader.requests_per_second = 1

    # Use the aload method to asynchronously load the documents
    # Note: You might need to handle exceptions or errors here
    try:
        data = loader.aload()
    except Exception as e:
        print(f"An error occurred: {e}")
        data = None

    return data


# In[7]:


import requests
from bs4 import BeautifulSoup
from requests import get

def read_urls_from_file(filepath):
    with open(filepath, 'r') as file:
        # Elimina comas y comillas extras de cada línea
        urls = [line.strip().rstrip(',').replace("'", "").replace('"', '') for line in file if line.strip()]
    return urls

# Assuming the 'urls' list is defined as containing the initial URLs you want to scrape
urls = read_urls_from_file('C:/users/julia/desktop/urls.txt') # List of URLs you've already defined
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0'}
base_url = 'https://www.dia.es'
all_unique_links = set()

for url in urls:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        category_segment = url.split('/')[-2]
        # Extract and adjust links to ensure they are absolute
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/'):
                href = base_url + href  # Converts relative links to absolute
            # Check if URL does not contain 'sort' in the last segment
            if href.startswith(base_url) and category_segment in href and 'sort' not in href.split('/')[-1]:
                all_unique_links.add(href)

# Convert set to list (if needed)
all_unique_links = list(all_unique_links)


# In[ ]:


len(all_unique_links)


# ### Cargar datos Mercadona

# In[ ]:


from langchain_community.document_loaders import TextLoader

def desde_txt(file_path):

    loader = TextLoader(file_path)
    try:
        data = loader.load()
    except Exception as e:
        print(f"An error occurred: {e}")
        data = None
    return data


# In[ ]:


def cargar_y_fragmentar(documento):
    """Carga el contenido desde la web o un archivo y lo fragmenta."""
    if isinstance(documento, str) and documento.endswith('.txt'):
        contenido = desde_txt(documento)
    else:
        contenido = desde_web2(documento)
    
    fragmentos = fragmentar(contenido)
    costo_embedding(fragmentos)
    return fragmentos


# ### Fragmentar los datos

# In[ ]:


def fragmentar(data, chunk_size=500):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=60)
    fragmentos = text_splitter.split_documents(data)
    return fragmentos


# ### Costos OpenAI

# In[ ]:


def costo_embedding(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0001:.5f}')


# ### Borrando Index de Pinecone

# In[ ]:


def borrar_indices(index_name='todos'):
    import pinecone
    pc = Pinecone(api_key="9aa9eec2-6465-43d6-a740-529299b29f77")

    if index_name == 'todos':
        indexes = pinecone.list_indexes()
        print('Borrando todos los índices ... ')
        for index in indexes:
            pc.delete_index(index)
        print('Listo!')
    else:
        print(f'Borrando el índice: {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Listo')


# In[ ]:


#borrar_indices("dia")


# ### Creando Vectores (Embeddings) y subirlos a (Pinecone)

# In[11]:


from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

api_key = "9aa9eec2-6465-43d6-a740-529299b29f77"

def creando_vectores(index_name):

    #embeddings = HuggingFaceEmbeddings()
    embeddings = OpenAIEmbeddings(api_key="sk-proj-PTpCD6bZUIFQ1QSGTJ39T3BlbkFJdxUmGAOYbg36BU18YcxC", model = 'text-embedding-ada-002')



    pc = Pinecone(api_key="9aa9eec2-6465-43d6-a740-529299b29f77")

    if index_name in pc.list_indexes().names():
        index = pc.Index(index_name)
        print(f'El índice {index_name} ya existe. Cargando los embeddings ... ', end='')
        vectores = PineconeVectorStore.from_existing_index(index_name, embeddings)

        
    elif index_name not in pc.list_indexes().names():
        
        documento_dia = all_unique_links  # Asumimos que all_unique_links es una lista de URLs
        fragmentos_dia = cargar_y_fragmentar(documento_dia)
        costo_embedding(fragmentos_dia)
        
        documento_mercadona = "mercadonafinal.txt"
        fragmentos_mercadona = cargar_y_fragmentar(documento_mercadona)
        costo_embedding(fragmentos_mercadona)

        print(f'Creando el índice {index_name} y los embeddings ...', end='')
        pc.create_index(name=index_name,dimension=1536,metric="cosine", spec=ServerlessSpec(cloud="aws",region="us-east-1"))
        vectores = PineconeVectorStore.from_documents(fragmentos, embeddings, index_name=index_name)
        
        
    return vectores


# ### Haciendo consultas

# In[68]:


import bs4
from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MergerRetriever,
)
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import (
    PineconeHybridSearchRetriever,
)
from prettytable import PrettyTable

# Inicializaciones necesarias
llm = ChatOpenAI(model='gpt-4o', api_key="sk-proj-PTpCD6bZUIFQ1QSGTJ39T3BlbkFJdxUmGAOYbg36BU18YcxC")
embeddings = OpenAIEmbeddings(api_key="sk-proj-PTpCD6bZUIFQ1QSGTJ39T3BlbkFJdxUmGAOYbg36BU18YcxC", model='text-embedding-ada-002')
store = {}

def modelo_final(pregunta):
    
    
    vectores_dia = PineconeVectorStore.from_existing_index('dia', embeddings)
    retriever_dia = vectores_dia.as_retriever(search_type='similarity', search_kwargs={'k': 10})
    
    vectores_mercadona = PineconeVectorStore.from_existing_index('superdown', embeddings)
    retriever_mercadona = vectores_mercadona.as_retriever(search_type='similarity', search_kwargs={'k': 10})
    
    retriever= MergerRetriever(retrievers=[retriever_dia, retriever_mercadona])
    
    ### Contextualize question ###
    contextualize_q_system_prompt = """Considera la última pregunta del usuario y reformúlala para que pueda entenderse independientemente de cualquier contexto previo en el historial de chat. No respondas la pregunta; simplemente asegúrate de que esté bien formulada y clara."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    ### Answer question ###
    qa_system_prompt = """Basándote en la información recuperada de los supermercados, proporciona una respuesta detallada en formato de tabla comparando los datos relevantes de DIA y Mercadona. Si la información es insuficiente para una respuesta completa, indica claramente cualquier limitación en los datos disponibles.
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    invoke_answer = conversational_rag_chain.invoke(
        {"input": pregunta},
        config={"configurable": {"session_id": "aid124"}}
    )["answer"]
    
    
    return invoke_answer


# In[69]:


while True:
    pregunta = input("Realiza una pregunta escribe 'salir' para terminar: \n")
    if pregunta.lower() == "salir":
        print("Adiós!!!")
        break
    else:
        respuesta = modelo_final(pregunta)
        print(respuesta)



# In[ ]:


pip install pyqt5


# In[ ]:


pip install pyqtwebengine


# In[ ]:


pip install -U SQLAlchemy


# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  pinecone-client pinecone-text')


# In[61]:


pip install prettytable


# In[73]:


import streamlit as st
from chatbot_model import modelo_final  # Asegúrate de que el nombre sea correcto

# Configuración inicial de Streamlit
st.set_page_config(page_title="Chatbot Inteligente")

# Proceso para manejar la entrada del usuario y mostrar la respuesta del bot
user_input = st.text_input("Escribe tu pregunta aquí:")
if user_input:
    response = modelo_final(user_input)
    st.write("Respuesta del bot:", response)



# In[ ]:




