import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Streamlit UI
st.title("Blog Post Generator")
st.markdown(
    """
    Enter a topic, and this app will generate a complete blog post for you:
    1. Summarizes the topic into 3 bullet points.
    2. Expands each bullet point into detailed paragraphs.
    3. Creates a polished final blog post with an introduction, body, and conclusion.
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

# Button to generate the blog post
if st.button("Generate Blog Post"):
    if not api_key.strip():
        st.error("OpenAI API key is required!")
    elif not topic.strip():
        st.error("Please enter a valid topic.")
    else:
        # Set OpenAI API key for LangChain
        try:
            from langchain.chat_models import ChatOpenAI
            import openai
            openai.api_key = api_key  # User-provided API key
            
            # Initialize the OpenAI Chat Model
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

            # Define Prompts and Chains
            summary_prompt = PromptTemplate(
                input_variables=["topic"],
                template="""
You are a writer tasked with summarizing a topic into 3 clear and concise bullet points.
Topic: {topic}
"""
            )
            expand_prompt = PromptTemplate(
                input_variables=["bullets"],
                template="""
Expand the following bullet points into detailed paragraphs for a blog post:
{bullets}
"""
            )
            final_output_prompt = PromptTemplate(
                input_variables=["expanded_blog"],
                template="""
You are a professional blog writer. Using the following expanded paragraphs, craft a complete and engaging blog post.

- Write a catchy introduction that introduces the topic and grabs the reader's attention.
- Use the expanded paragraphs as the main body of the blog post. Organize them logically.
- Conclude the blog post with a compelling summary and call-to-action if relevant.

Expanded paragraphs:
{expanded_blog}

Now, write the full blog post:
"""
            )

            # Create LangChain Chains
            summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="bullets")
            expand_chain = LLMChain(llm=llm, prompt=expand_prompt, output_key="expanded_blog")
            output_chain = LLMChain(llm=llm, prompt=final_output_prompt, output_key="blog_post")

            # Generate Content
            with st.spinner("Generating bullet points..."):
                bullets = summary_chain.run({"topic": topic})
                st.subheader("Step 1: Bullet Points")
                st.write(bullets)

            with st.spinner("Expanding bullet points into paragraphs..."):
                expanded_blog = expand_chain.run({"bullets": bullets})
                st.subheader("Step 2: Expanded Paragraphs")
                st.write(expanded_blog)

            with st.spinner("Generating the final blog post..."):
                blog_post = output_chain.run({"expanded_blog": expanded_blog})
                st.subheader("Step 3: Final Blog Post")
                st.write(blog_post)

            st.success("Blog post generation complete!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
