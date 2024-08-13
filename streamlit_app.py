import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# Load the models
@st.cache_resource
def load_models():
    # Loading the necessary models
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    generator = pipeline("text-generation", model="gpt2")
    sentiment_analyzer = pipeline("sentiment-analysis")
    question_answerer = pipeline("question-answering")
    
    # Load the Stable Diffusion pipeline for image generation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_generator = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    image_generator.to(device)
    
    return summarizer, generator, sentiment_analyzer, question_answerer, image_generator

# Initialize Streamlit app
st.title("Multifunctional NLP and Image Generation Tool")

# Load the models
summarizer, generator, sentiment_analyzer, question_answerer, image_generator = load_models()

# Task selection
task = st.sidebar.selectbox(
    "Choose a task", 
    ("Text Summarization", "Next Word Prediction", "Story Prediction", "Chatbot", "Sentiment Analysis", "Question Answering", "Image Generation")
)

# Task: Text Summarization
if task == "Text Summarization":
    st.header("Text Summarization")
    text = st.text_area("Enter text to summarize")
    if st.button("Summarize"):
        summary = summarizer(text)[0]['summary_text']
        st.write(summary)

# Task: Next Word Prediction
elif task == "Next Word Prediction":
    st.header("Next Word Prediction")
    text = st.text_input("Enter the beginning of a sentence")
    if st.button("Predict"):
        continuation = generator(text, max_length=30)[0]['generated_text']
        st.write(continuation)

# Task: Story Prediction
elif task == "Story Prediction":
    st.header("Story Prediction")
    prompt = st.text_input("Enter a story prompt")
    if st.button("Generate Story"):
        story = generator(prompt, max_length=100)[0]['generated_text']
        st.write(story)

# Task: Chatbot
elif task == "Chatbot":
    st.header("Chatbot")
    user_input = st.text_input("You: ")
    if st.button("Send"):
        response = generator(user_input, max_length=50)[0]['generated_text']
        st.write(f"Bot: {response}")

# Task: Sentiment Analysis
elif task == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    text = st.text_area("Enter text for sentiment analysis")
    if st.button("Analyze Sentiment"):
        sentiment = sentiment_analyzer(text)[0]
        st.write(f"Label: {sentiment['label']}, Score: {sentiment['score']}")

# Task: Question Answering
elif task == "Question Answering":
    st.header("Question Answering")
    question = st.text_input("Enter your question")
    context = st.text_area("Enter the context")
    if st.button("Get Answer"):
        answer = question_answerer(question=question, context=context)['answer']
        st.write(answer)

# Task: Image Generation
elif task == "Image Generation":
    st.header("Image Generation")
    prompt = st.text_input("Enter a prompt for image generation")
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            image = image_generator(prompt).images[0]
            st.image(image, caption="Generated Image")
