from sentence_transformers import SentenceTransformer
import pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)
import streamlit as st
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='56a9ce45-5652-4776-80a8-db04c55ea8dd', environment='gcp-starter')
index = pinecone.Index('hanhwa-chatbot')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']