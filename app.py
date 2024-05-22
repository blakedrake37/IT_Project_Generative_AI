import streamlit as st
from llm_chains import load_normal_chain, load_pdf_chat_chain, searchYoutube, searchWiki
from utils import get_timestamp, load_config, get_avatar
from pdf_handler import add_documents_to_db
from html_templates import css
from database_operations import load_last_k_text_messages, save_text_message, load_messages, get_all_chat_history_ids, delete_chat_history
import sqlite3



config = load_config()

@st.cache_resource
def load_chain(model_name):
    if st.session_state.pdf_chat:
        print("loading pdf chat chain")
        return load_pdf_chat_chain(model_name)
    return load_normal_chain(model_name)

def toggle_pdf_chat():
    st.session_state.pdf_chat = True
    clear_cache()

def get_session_key():
    if st.session_state.session_key == "new_session":
        st.session_state.new_session_key = get_timestamp()
        return st.session_state.new_session_key
    return st.session_state.session_key

def delete_chat_session_history():
    delete_chat_history(st.session_state.session_key)
    st.session_state.session_index_tracker = "new_session"

def clear_cache():
    st.cache_resource.clear()

def main():
    st.set_page_config(page_title="Chatbot", layout='wide', page_icon="💭")
    st.title("Generative AI App")
    st.write(css, unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <style>
        .reportview-container .main footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <footer>
        <p> IT Project - Generative AI App</p>
        <p> Ông Vũ Hữu Tài - 21110796 🧑‍🎓</p>
        <p> Lê Nguyễn Toàn Tâm - 21110797 👩‍🎓</p>
        </footer>
        """,
        unsafe_allow_html=True,
    )

    if "db_conn" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.db_conn = sqlite3.connect(config["chat_sessions_database_path"], check_same_thread=False)
        st.session_state.audio_uploader_key = 0
        st.session_state.pdf_uploader_key = 1
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None
    st.sidebar.title("Chat Models")
    models = {
        "Mistral",
        "Search Youtube",
        "Search Wikipedia",
        "Vistral"
    }
    st.sidebar.selectbox("Select a chat model", models, key="model_key")
    model_name = st.session_state.model_key
    if model_name=="Mistral":
        st.sidebar.caption("It's not supported Vietnamese")
    elif model_name=="Vistral":
        st.sidebar.caption("It's supported Vietnamese")
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + get_all_chat_history_ids()

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)

    
    delete_chat_col, clear_cache_col = st.sidebar.columns(2)
    delete_chat_col.button("Delete Chat Session", on_click=delete_chat_session_history)
    clear_cache_col.button("Clear Cache", on_click=clear_cache)
    if model_name == "Mistral" or model_name == "Vistral":
        pdf_toggle_col, voice_rec_col = st.sidebar.columns(2)
        pdf_toggle_col.toggle("PDF Chat", key="pdf_chat", value=False, on_change=clear_cache)
    chat_container = st.container()
    user_input = st.chat_input("Type your message here", key="user_input")
    if model_name == "Mistral" or model_name == "Vistral":
        uploaded_pdf = st.sidebar.file_uploader("Upload a pdf file", accept_multiple_files=True, 
                                                key=st.session_state.pdf_uploader_key, type=["pdf"], on_change=toggle_pdf_chat)
    st.sidebar.caption(
            'The project for IT project in [HCMUTE](https://hcmute.edu.vn) 2023-2024')
    if model_name == "Mistral" or model_name == "Vistral":
        if uploaded_pdf:
            with st.spinner("Processing pdf..."):
                model_name = st.session_state.model_key
                add_documents_to_db(uploaded_pdf, model_name)
                st.session_state.pdf_uploader_key += 2




    
    if user_input:
        model_name = st.session_state.model_key
        if model_name=="Search Youtube":
            tool = searchYoutube()
            llm_answer = tool.run(user_input)
        elif model_name == "Search Wikipedia":
            wikipedia  = searchWiki()
            llm_answer = wikipedia.run(user_input)
        else:
            llm_chain = load_chain(model_name)
            llm_answer = llm_chain.run(user_input = user_input, 
                                        chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"]))
        save_text_message(get_session_key(), "human", user_input)
        save_text_message(get_session_key(), "ai", llm_answer)
        user_input = None


    if (st.session_state.session_key != "new_session") != (st.session_state.new_session_key != None):
        with chat_container:
            chat_history_messages = load_messages(get_session_key())

            for message in chat_history_messages:
                with st.chat_message(name=message["sender_type"], avatar=get_avatar(message["sender_type"])):
                    if message["message_type"] == "text":
                        st.write(message["content"])



        if (st.session_state.session_key == "new_session") and (st.session_state.new_session_key != None):
            st.rerun()
        


if __name__ == "__main__":
    main()