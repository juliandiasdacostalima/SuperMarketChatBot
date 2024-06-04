#!/usr/bin/env python

import os
import bs4
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain.chains import ConversationalRetrievalChain,create_history_aware_retriever, create_retrieval_chain
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
    PineconeHybridSearchRetriever
)
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv(find_dotenv(), override=True

# Obtener las variables de entorno
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicializaciones necesarias
llm = ChatOpenAI(model='gpt-4o', api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-ada-002')
store = {}

def modelo_final(pregunta):
    vectores_dia = PineconeVectorStore.from_existing_index('dia', embeddings, api_key=PINECONE_API_KEY)
    retriever_dia = vectores_dia.as_retriever(search_type='similarity', search_kwargs={'k': 10})

    vectores_mercadona = PineconeVectorStore.from_existing_index('superdown', embeddings, api_key=PINECONE_API_KEY)
    retriever_mercadona = vectores_mercadona.as_retriever(search_type='similarity', search_kwargs={'k': 10})

    retriever = MergerRetriever(retrievers=[retriever_dia, retriever_mercadona])

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
    qa_system_prompt = """Analiza y compara la información obtenida de los supermercados Mercadona y DIA. Proporciona tus hallazgos en dos tablas separadas: una para Mercadona y otra para DIA. Cada tabla debe incluir comparaciones detalladas de los productos en cuanto a precios, calidad y disponibilidad. Si los datos son insuficientes para una comparación exhaustiva, identifica los productos que son más económicos en cada supermercado y señala cualquier limitación en la información disponible. Asegúrate de que tu respuesta sea clara y bien organizada para facilitar la comprensión.
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
        config={"configurable": {"session_id": "aid125"}}
    )["answer"]

    return invoke_answer
