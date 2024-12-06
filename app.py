import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
import base64

# Streamlit UI
st.title("Blog Post Generator with AI-Generated Background")

st.markdown(
    """
    Enter a topic, and this app will:
    1. Generate a complete blog post.
    2. Create a visually appealing background image based on the topic.
    """
)

# Input for OpenAI API Key
api_key = st.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
)

# Check for valid API key
if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")

# Input for the topic
topic = st.text_input(
    "Enter the topic for your blog post:", 
    placeholder="E.g., The importance of sustainable energy"
)

# Generate button
if st.button("Generate Blog Post and Background"):
    if not api_key.strip():
        st.error("OpenAI API key is required!")
    elif not topic.strip():
        st.error("Please enter a valid topic.")
    else:
        try:
            # Initialize OpenAI API
            openai.api_key = api_key
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=api_key)

            # Define Prompts and Chains
            summary_prompt = PromptTemplate(
                input_variables=["topic"],
                template="Summarize the topic into 3 bullet points. Topic: {topic}"
            )
            expand_prompt = PromptTemplate(
                input_variables=["bullets"],
                template="Expand the following bullet points into detailed paragraphs: {bullets}"
            )
            final_output_prompt = PromptTemplate(
                input_variables=["expanded_blog"],
                template="Write a complete blog post based on these expanded paragraphs: {expanded_blog}"
            )
            summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
            expand_chain = LLMChain(llm=llm, prompt=expand_prompt)
            output_chain = LLMChain(llm=llm, prompt=final_output_prompt)

            # Generate Blog Post
            with st.spinner("Generating bullet points..."):
                bullets = summary_chain.run({"topic": topic})
                st.subheader("Step 1: Bullet Points")
                st.write(bullets)

            with st.spinner("Expanding bullet points..."):
                expanded_blog = expand_chain.run({"bullets": bullets})
                st.subheader("Step 2: Expanded Paragraphs")
                st.write(expanded_blog)

            with st.spinner("Creating final blog post..."):
                blog_post = output_chain.run({"expanded_blog": expanded_blog})
                st.subheader("Step 3: Blog Post")
                st.write(blog_post)

            # Generate Background Image
            with st.spinner("Generating background image..."):
                dalle_prompt = f"Create a visually appealing and thematic background image based on the topic: {topic}."
                response = openai.Image.create(
                    prompt=dalle_prompt,
                    n=1,
                    size="1024x1024"
                )
                image_url = response['data'][0]['url']

                # Display the background image
                st.markdown(
                    f"""
                    <style>
                    .stApp {{
                        background-image: url({image_url});
                        background-size: cover;
                        background-position: center;
                        background-repeat: no-repeat;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.success("Background image generated!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
