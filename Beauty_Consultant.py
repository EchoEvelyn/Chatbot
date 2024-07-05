#pip install streamlit langchain openai faiss-cpu tiktoken

import tempfile
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
# from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS

import pandas as pd
from PIL import Image
import openai

import os
os.environ["OPENAI_API_KEY"] = "sk-proj-cU24cPDmI3NA5AMpMObVT3BlbkFJYy1add2zcoLAX2FnRBjO"
openai.api_key  = os.getenv('OPENAI_API_KEY')

image = Image.open('background.jpeg')
st.sidebar.image(image)

st.sidebar.title(
    'Welcome to Beauty Consultant :kiss: ~'
)
st.sidebar.subheader(
    ":triangular_flag_on_post: How to use?"
)
st.sidebar.markdown("- Upload a file")
st.sidebar.markdown("- Ask questions about the file")
st.sidebar.markdown("- Add to your shopping cart")

st.sidebar.write("---")

api_key_input = st.sidebar.text_input(
    label="Your OpenAI API Key 🔑",
    type="password",
    placeholder="Enter your OpenAI API key here (sk-...)",
    help="You can get your API key from https://platform.openai.com/account/api-keys.",
)

st.session_state["OPENAI_API_KEY"] = api_key_input

uploaded_file = st.file_uploader(label=" ", type="csv")

if not api_key_input:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()
else:

    if uploaded_file :

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding='unicode_escape')
            data = loader.load()
        finally:
            os.remove(tmp_file_path)

        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(data, embeddings)

        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=api_key_input),
            retriever=vectors.as_retriever()
        )

        tab1, tab2, tab3 = st.tabs(["💬Conversation", "📔Data", "🛒Shopping Cart"])

        def conversational_chat(query):
            if "add" in query:
                with tab3:
                    st.text("")
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


        def extract_products(response): # extract product names from a response
            prompt = f"""
            Extract the brands and names of makeup products from the below paragraph and present them with comma separation.\
            If there is no product extracted, return no product recommended.\
            If there is product extracted, no need to type words like product recommended.

            paragraph: ```{response}```
            """
            extraction = get_completion(prompt)
            return extraction


        if 'input_query' not in st.session_state: # user's input
            st.session_state['input_query'] = ["Hey ! 👋"]

        if 'response' not in st.session_state: # chatbot's response
            st.session_state['response'] = ["Hi, I'm your beauty consultant. How can I help you today?"]

        if 'history' not in st.session_state: # overal conversation
            st.session_state['history'] = []

        if 'shopping_cart' not in st.session_state:
            st.session_state['shopping_cart'] = []

        with tab1:
            st.header('Chatbot')
            st.markdown(":pushpin: Add feature:")
            st.markdown("- If you want to add specific products from the recommendations, type: add + names")
            st.markdown("- If you want to add all products in the recommendations, type: add")
            #container for the chat history
            response_container = st.container()
            #container for the user's text input
            input_container = st.container()

            with input_container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_input("Question:", placeholder="What questions do you have ?", key='input')
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    index = user_input.casefold().find('add')
                    if index != -1:
                        w_nw_products = extract_products(user_input)
                        index_2 = w_nw_products.casefold().find('no product recommended')
                        if index_2 != -1:
                            product_list = extract_products(st.session_state['response'][-1])
                        else:
                            product_list = extract_products(user_input)
                        st.session_state['shopping_cart'].append(product_list)
                        output = "Great! The item will be shown once you go to the shopping cart."
                    else:
                        output = conversational_chat(user_input)
                    st.session_state['input_query'].append(user_input)
                    st.session_state['response'].append(output)

            if st.session_state['response']:
                with response_container:
                    for i in range(len(st.session_state['response'])):
                        message(st.session_state["input_query"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                        message(st.session_state["response"][i], key=str(i), avatar_style="thumbs")

        with tab2:
            st.header('Data Statistics')
            df = pd.read_csv(uploaded_file)
            st.write(df.describe())
            st.header('Data Header')
            st.write(df.head())

        with tab3:
            st.header('Your Shopping Cart')
            clear_button = st.button("Clear All")
            placeholder = st.empty()
            with placeholder.container():
                for each in st.session_state.shopping_cart:
                    index = each.find(',')
                    if index != -1:
                        split = each.split(",")
                        for i in split:
                            st.markdown("- " + i + "\n")
                    else:
                        st.markdown("- " + each + "\n")
            if clear_button:
                placeholder.empty()
                st.session_state.shopping_cart = []
