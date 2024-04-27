import re
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()




client = chromadb.PersistentClient(path="E:/chroma2")

collection = client.get_collection(name="search_engine")

collection_2=client.get_collection(name="search_engine_1")

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def preprocess(text):
    # Removing special characters and digits
    sentence = re.sub("[^a-zA-Z]", " ", text)

    # change sentence to lower case
    sentence = sentence.lower()

    # tokenize into words
    tokens = sentence.split()

    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def encode_text(text):
    embedding=model.encode(text)
    return embedding 

def remove(data):
    only_ids = [re.sub('/.*$', '', s) for s in data]
    return only_ids

st.header("Movie Subtitles-Search Engine!!!")
query=st.text_input("Enter  a subtitle..")
if st.button("Search")==True:
    st.subheader("Top 10 similar search titles:")
    clean_query=preprocess(query)
    embd_query = model.encode(clean_query).tolist()
    query1=collection.query( query_embeddings = embd_query, n_results=10) 
    ele=query1['ids'][0] 
    lst_ids=remove(ele) 

    for item in lst_ids:

        response=collection_2.get(ids=item)
        st.write(response['documents'][0])



    







