import os
import pandas as pd

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader


class VectorStore:
    def __init__(self, rag_cfg):
        self.file_path = rag_cfg.data_root
        self.enrollments = pd.read_csv(f'{self.file_path}/enrollments.csv')
        self.to_data_list = self.enrollments[self.enrollments['status'] == 'new']['name'].tolist()

        self.embeddings = OpenAIEmbeddings(api_key=rag_cfg.api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
        )

        self.persist_directory = rag_cfg.persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

    def update(self):
        for data_name in self.to_data_list:
            data = self.get_data(data_name)
            self.create_vector_store(data, data_name)

    def get_data(self, data_name):
        data_path = f'{self.file_path}/raw_data/{data_name}'
        loader = PyMuPDFLoader(data_path)
        docs = loader.load()
        data = self.text_splitter.split_documents(docs)
        return data

    def create_vector_store(self, data, store_name):
        if not data:
            print(f"-----No data to create vector store for {store_name}-----")
            return

        persistent_directory = f'{self.persist_directory}/{store_name}'
        if not os.path.exists(persistent_directory):      
            vector_store = Chroma.from_documents(data, self.embeddings)
            print(f"-----Created vector store for {store_name}-----")
            self.enrollments.loc[self.enrollments['name'] == store_name, 'status'] = 'enrolled'

        else:
            print(f"-----Vector store for {store_name} already exists-----")

        
    def query_vector_store(self):
        if os.path.exists(self.persist_directory):
            print('-----Querying vector store-----')
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            
            retriever = db.as_retriever(
                search_type='similarity',
                search_kwargs={'k': 5}
            )

            return retriever

        else:
            print('-----Vector store does not exist-----')

