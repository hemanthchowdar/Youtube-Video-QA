import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai


key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=key)


llm = genai.GenerativeModel('gemini-2.5-flash-lite')

def call_gemini(prompt_value):
    return llm.generate_content(prompt_value.to_string()).text

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=key, model="models/gemini-embedding-001")



def get_transcript(video_id):
    """Fetch YouTube video transcript"""
    try:
        print(video_id)
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id).to_raw_data()
        return " ".join(chunk['text'] for chunk in transcript_list)
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None


def create_retriever(text):
    """Create vector store retriever from text"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

def extract_video_id(youtube_link):
    """Extract video ID from YouTube URL"""
    if "youtube.com" in youtube_link:
        parsed_url = urlparse(youtube_link)
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    elif "youtu.be" in youtube_link:
        parsed_url = urlparse(youtube_link)
        return parsed_url.path.lstrip("/")
    return None

# Streamlit UI
st.title("YouTube Video Q&A Assistant")

youtube_link = st.text_input("Enter YouTube Link:")

if youtube_link:
    video_id = extract_video_id(youtube_link)
    
    if video_id:
        # Display video
        st.subheader('Video')
        st.markdown(
            f'<iframe width="640" height="360" src="https://www.youtube.com/embed/{video_id}" '
            f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
            f'allowfullscreen></iframe>',
            unsafe_allow_html=True
        )
        
        user_question = st.text_area("Ask a question about the video:")
        
        if st.button("Send"):
            if user_question:
                with st.spinner("Processing..."):
                    # Get transcript and create retriever
                    transcript = get_transcript(video_id)
                    
                    if transcript:
                        retriever = create_retriever(transcript)
                        
                        # Create prompt
                        prompt = PromptTemplate(
                            template="""You are a helpful assistant. Answer the question based on the provided context from the YouTube video.
                            
Context: {context}

Question: {question}

Answer:""",
                            input_variables=['context', 'question']
                        )
                        
                        # Build chain
                        format_docs = lambda x: "\n\n".join(doc.page_content for doc in x)
                        
                        chain = (
                            RunnableParallel({
                                'context': retriever | RunnableLambda(format_docs),
                                'question': RunnablePassthrough()
                            })
                            | prompt 
                            | RunnableLambda(call_gemini) 
                            | StrOutputParser()
                        )
                        
                        # Get answer
                        result = chain.invoke(user_question)
                        st.info("ℹ️ Answer is based on the YouTube video transcript.")
                        st.write(result)
            else:
                st.warning("Please enter a question.")
    else:
        st.error("Invalid YouTube link. Please check the URL.")