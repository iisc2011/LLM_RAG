import streamlit as st
from chat_with_documents import configure_retrieval_chain
from utils import MEMORY, DocumentLoader
from streamlit.external.langchain import StreamlitCallbackHandler

import logging

logging.basicConfig(encoding='utf-8', level=logging.INFO)
LOGGING = logging.getLogger()

st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.title("🦜 LangChain: Chat with Documents")

uploaded_files = st.sidebar.file_uploader(
    label='upload_files',
    type= list(DocumentLoader.supported_extension.keys()),
    accept_multiple_files=True
)

if not uploaded_files:
    st.info('Please upload the documents')
    st.stop()

use_compression = st.checkbox('Compression', value=False)
use_moderation = st.checkbox('Moderation', value=False)

LOGGING.info('use_compression', use_compression)
LOGGING.info('use_moderation', use_moderation)


conversional_chain = configure_retrieval_chain(
    files=uploaded_files,
    use_compression=use_compression,
     use_moderation=use_moderation
)


LOGGING.info('response from conversional_chain')

avatars = {"human": "user", "ai": "assistant"}

if st.sidebar.button('clear message history'):
    MEMORY.chat_memory.clear()

if len(MEMORY.chat_memory.messages) == 0:
    st.chat_message('assistant').markdown('Ask me ')

for msg in MEMORY.chat_memory.messages:
    st.chat_message(name=avatars[msg.type],avatar=avatars[msg.type]).write(msg.content)   

assistant = st.chat_message('assistant')

if user_query := st.chat_input(placeholder='Give me 3 keywords for what you have right now'):
    st.chat_message('user').write(user_query)
    container = st.empty()
    stream_handler = StreamlitCallbackHandler(container)
    with st.chat_message('assistant'):
        LOGGING.info('conversional_chain.run....')
        response = conversional_chain.run(
            {
            'question': user_query,
            'chat_history': MEMORY.chat_memory.messages
            },callbacks=[stream_handler]
        )
        
        #LOGGING.info('response', response)
        if response:
            try:

                container.markdown(response)
            except Exception as e:
                
                st.write(f'Something wrong: {e}')   
                st.stop() 
            





