# IaChatbotPdf

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  #OpenAIS embeddings (incorporação de vetor para armazenar no banco de dados)
from langchain.vectorstores import FAISS

 
#Configurando a leitura do pdf pagina por pagina
def get_pdf_text(pdf_docs):
   text = ""
   for pdf in pdf_docs:
       pdf_reader = PdfReader(pdf)
       for page in pdf_reader.pages:
           text += page.extract_text()
   return text      
 
#Criando leitura de pedaços do texto para alimentar o banco de dados"
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
       separator="\n",
       chunk_size=1000,
       chunk_overlap=200,
       length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
 
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
 
def main():
  #para acessar as keys ( openai e hugginface)
  load_dotenv()
 
  #titulo da interface e icone
  st.set_page_config(page_title="Chat com multiplos PDFs", page_icon=":books:")
 
  #cabeçalho
  st.header("Chat com multiplos PDFs :books:")
 
  #local para escrever a pergunta
  st.text_input("Escreva sua pergunta sobre a documentação:")
 
  #local para fazer upload dos arquivos (sidebar)
  with st.sidebar:
    st.subheader("Seus documentos")
    pdf_docs = st.file_uploader("Upload seus PDFs aqui e clique para 'Subir'", accept_multiple_files=True)
    if st.button("Subir"):
      with st.spinner("Processando"):
         #coletando texto pdf
         raw_text = get_pdf_text(pdf_docs)       
         #coletando pedaços do texto
         text_chunks = get_text_chunks(raw_text)
         #coletando vetor de armazenamento(base de dados)
         vectorstore = get_vectorstore(text_chunks)
 
 
if __name__ == '__main__':
       main()
