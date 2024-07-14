import tempfile
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from PIL import Image
import openai
import pandas as pd
# import plotly.express as px

import os

image = Image.open('background.jpg')
st.sidebar.image(image)

st.sidebar.title(
    'Welcome to Virtual Reader :memo:'
)

st.sidebar.markdown("""
                    :triangular_flag_on_post: Enter your API Key and upload your file~
                    Feel free to ask me any questions about the file~
                    """)

st.sidebar.write("---")

api_key_input = st.sidebar.text_input(
    label="Your OpenAI API Key 🔑",
    type="password",
    placeholder="Enter your OpenAI API key here (sk-...)",
    help="You can get your API key from https://platform.openai.com/account/api-keys.",
)

st.session_state["OPENAI_API_KEY"] = api_key_input
os.environ["OPENAI_API_KEY"] = api_key_input

if not api_key_input:
    st.info("Please enter a valid API key to continue.")
    st.stop()
else:
    uploaded_file = st.file_uploader(label=" ", type="csv")

    if uploaded_file :

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding='unicode_escape')
            data = loader.load()
            df = pd.read_csv(tmp_file_path, encoding='unicode_escape')
        finally:
            os.remove(tmp_file_path)

        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(data, embeddings)

        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=api_key_input),
            retriever=vectors.as_retriever()
        )

        def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        def get_completion(prompt, model="gpt-3.5-turbo"):
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message["content"]

        if 'input_query' not in st.session_state: # user's input
            st.session_state['input_query'] = ["Hey ! 👋"]

        if 'response' not in st.session_state: # chatbot's response
            st.session_state['response'] = ["Hi~ How can I help you today?"]

        if 'history' not in st.session_state: # overal conversation
            st.session_state['history'] = []

        # tab1 = st.tabs('💬Conversation')

        # with tab1:
        #container for the chat history
        response_container = st.container()
        #container for the user's text input
        input_container = st.container()
        with input_container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Question:", placeholder="What questions do you have ?", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['input_query'].append(user_input)
                st.session_state['response'].append(output)

        if st.session_state['response']:
            with response_container:
                for i in range(len(st.session_state['response'])):
                    message(st.session_state["input_query"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["response"][i], key=str(i), avatar_style="thumbs")

        # with tab2:
        #     st.write(df.sample(10))

        #     st.write("### Plot Customization")

        #     columns = df.columns.tolist()
        #     x_axis = st.selectbox("X-Axis:", [None] + columns)
        #     y_axis = st.selectbox("Y-Axis:", [None] + columns)
        #     color = st.selectbox("Color:", [None] + columns)

        #     plot_title = st.text_input("Plot Title:", value=f"{x_axis} vs {y_axis}")
        #     plot_type = st.selectbox("Plot Type:", ['scatter', 'line', 'bar'])

        #     # Main panel for the plot
        #     if x_axis and y_axis:
        #         if plot_type == 'scatter':
        #             fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=plot_title)
        #         elif plot_type == 'line':
        #             fig = px.line(df, x=x_axis, y=y_axis, color=color, title=plot_title)
        #         elif plot_type == 'bar':
        #             fig = px.bar(df, x=x_axis, y=y_axis, color=color, title=plot_title)

        #         st.plotly_chart(fig)
        #     else:
        #         st.write("Please select both X and Y axis columns.")
