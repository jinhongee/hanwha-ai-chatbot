from langchain.document_loaders import DirectoryLoader

directory = './data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
print("documents,", len(documents))


from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=1000,chunk_overlap=100):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print("docs,", len(docs))

print(docs[8].page_content)


from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

import pinecone 
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key="56a9ce45-5652-4776-80a8-db04c55ea8dd",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
)
index_name = "hanhwa-chatbot"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

def get_similiar_docs(query,k=1,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs

