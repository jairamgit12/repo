import streamlit as st
import langchain_helper

st.title("Sample Version of ChatGPT")

text_input = st.text_input("Search :")



if text_input:
    response = langchain_helper.about_the_topic(text_input)
    st.header(response['Topic'].strip())
    

    









