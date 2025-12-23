# -*- coding: utf-8 -*-
"""
NLP Application Suite PRO - Streamlit Web App Edition v2.6 (Ultimate Edition)
(Patched for Streamlit Cloud 1GB RAM Limit)
"""

import os
import re
import string
import shutil
import pandas as pd
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
import torch
import warnings
import sys
import random
import time
import emoji
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO

# Hide warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Library and NLTK Setup ---
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    for pkg in ("punkt", "stopwords", "wordnet"):
        try: nltk.data.find(f"tokenizers/{pkg}")
        except LookupError: nltk.download(pkg, quiet=True)
except ImportError:
    st.error("NLTK not found. Please run: pip install nltk")
    st.stop()

try:
    from spellchecker import SpellChecker
    SPELLCHECK_AVAILABLE = True
except ImportError:
    SPELLCHECK_AVAILABLE = False


try: from textblob import TextBlob
except ImportError: 
    st.error("TextBlob not found. Please run: pip install textblob")
    st.stop()

TRANSFORMERS_AVAILABLE = False
try:
    from transformers import (pipeline, AutoTokenizer, AutoModelForCausalLM,
                              Trainer, TrainingArguments, DataCollatorForLanguageModeling,
                              TrainerCallback)
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError: pass

# --- Page Config & Custom CSS ---
st.set_page_config(
    page_title="NLP Application Suite PRO",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

def local_css():
    st.markdown("""
    <style>
        @keyframes cosmic-flow {
            0%{background-position:0% 50%}
            50%{background-position:100% 50%}
            100%{background-position:0% 50%}
        }

        /* Base64 encoded seamless starfield image */
        .stars-bg {
            background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkAQMAAABKLAcXAAAABlBMVEUAAAUAAAB9goL0AAAAAXRSTlMAQObYZgAAAChJREFUOMtjYBgFo2AUjIJRMApGwSgYBaNgFIyCUTAKRsEwGgAACfsAAR3y0DkAAAAASUVORK5CYII=');
            animation: cosmic-flow 50s linear infinite;
        }

        [data-testid="stRoot"] > div:first-child {
            background-color: #0d0d2b;
            color: #e0e0e0;
        }

        .stApp {
            background-color: transparent;
        }
        
        .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label, .stSelectbox label, .stTextArea label, .stFileUploader label {
            color: #e0e0e0 !important;
        }
        
        [data-testid="stAppViewContainer"] > .main > div:first-child {
            background: rgba(15, 12, 41, 0.7);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(173, 216, 230, 0.2);
            box-shadow: 0 0 25px rgba(0, 191, 255, 0.2);
        }

        .stButton>button {
            border-radius: 20px;
            background: linear-gradient(90deg, #36d1dc, #5b86e5);
            color: #ffffff;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 0 15px rgba(54, 209, 220, 0.6);
        }
        .stButton>button:hover {
            color: #ffffff !important;
            transform: scale(1.05);
            box-shadow: 0 0 30px rgba(91, 134, 229, 0.8);
        }

        [data-testid="stSidebar"] > div:first-child {
            background: rgba(15, 12, 41, 0.7);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-right: 1px solid rgba(173, 216, 230, 0.2);
            box-shadow: 0 0 30px rgba(0, 0, 0, 0.2);
        }
        
        [data-testid="stSidebar"] .stRadio label p {
             color: #e0e0e0 !important;
        }
        
    </style>
    <div class="stars-bg"></div>
    """, unsafe_allow_html=True)

local_css()


# --- Core NLP Functions ---
CHAT_WORDS = {
    "lol": "laughing out loud", "brb": "be right back", "btw": "by the way",
    "omg": "oh my god", "imo": "in my opinion", "imho": "in my humble opinion",
    "ttyl": "talk to you later", "np": "no problem", "ty": "thank you",
    "idk": "i don't know", "rofl": "rolling on the floor laughing", "thx": "thanks"
}

def preprocess_corpus(text, options, progress_callback=None):
    if not isinstance(text, str): return ""
    
    stages = {
        "Cleaning HTML/URLs": (10, lambda t: BeautifulSoup(t, "html.parser").get_text() and re.sub(r'https?://\S+|www\.\S+', '', t) if options.get("Remove HTML/URLs") else t),
        "Handling Emojis": (5, lambda t: emoji.demojize(t, delimiters=(" ", " ")) if options.get("Handle Emojis") == "Convert to Text" else (emoji.replace_emoji(t, replace='') if options.get("Handle Emojis") == "Remove" else t)),
        "Expanding Chat Slang": (10, lambda t: ' '.join([CHAT_WORDS.get(word.lower(), word) for word in t.split()]) if options.get("Expand Chat Slang") else t),
        "Lowercasing": (5, lambda t: t.lower() if options.get("Lowercase") else t),
        "Tokenizing": (15, lambda t: word_tokenize(t)),
        "Identifying Misspelled Words": (30, lambda t: (t, spell.unknown(t)) if options.get("Spelling Correction") and SPELLCHECK_AVAILABLE and isinstance(t, list) and (spell := SpellChecker()) else (t, set())),
        "Applying Corrections": (15, lambda t_tuple: ([spell.correction(token) if token in t_tuple[1] else token for token in t_tuple[0]] if options.get("Spelling Correction") and SPELLCHECK_AVAILABLE and isinstance(t_tuple[0], list) and (spell := SpellChecker()) else t_tuple[0])),
        "Finalizing Tokens": (10, lambda t: ' '.join(filter_and_lemmatize(t, options)) if isinstance(t, list) else t)
    }

    total_weight = sum(w for w, f in stages.values())
    cumulative_weight = 0
    
    processed_data = text

    for stage_name, (weight, func) in stages.items():
        if progress_callback:
            progress_callback(cumulative_weight / total_weight, f"Processing: {stage_name}...")
        
        processed_data = func(processed_data)
        cumulative_weight += weight
        
    if progress_callback:
        progress_callback(1.0, "Finished")

    return processed_data


def filter_and_lemmatize(tokens, options):
    final_tokens = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    for token in tokens:
        if token is None: continue
        
        if not token.isalpha():
            if options.get("Remove Punctuation"):
                continue
            else: 
                pass
        
        if options.get("Remove Stopwords") and token.lower() in stop_words:
            continue
            
        if options.get("Lemmatize"):
            final_tokens.append(lemmatizer.lemmatize(token))
        else:
            final_tokens.append(token)
            
    return final_tokens

def get_corpus_context(corpus, num_sentences=3):
    """Extracts a random snippet from the corpus to use as context."""
    if not corpus: return ""
    sentences = sent_tokenize(corpus)
    return ' '.join(random.sample(sentences, min(len(sentences), num_sentences))) if sentences else ""

@st.cache_resource
def load_nlp_pipeline(model_name, task):
    """
    Loads a single NLP pipeline and caches it.
    This is now called on-demand (when a button is clicked)
    to save memory on Streamlit Cloud.
    """
    try:
        return pipeline(task, model=model_name)
    except Exception as e:
        st.error(f"Error loading model '{model_name}' for task '{task}'. Please check model name and connection. Error: {e}")
        return None

# --- UI Functions ---
#
# DELETED 'load_pipelines_with_progress' function.
# It was loading all models at once and crashing the app
# due to the 1GB RAM limit on Streamlit Cloud.
#
# --- Streamlit UI ---

# Initialize session state
if 'suite_mode' not in st.session_state:
    st.session_state.suite_mode = "Light"
    st.session_state.corpus = None
    st.session_state.processed_corpus = None
    st.session_state.base_model_name = "distilgpt2"
    st.session_state.epochs = 3
    st.session_state.app_stage = "welcome"
    st.session_state.pro_models_loaded = False
    st.session_state.fine_tuned_model_path = None


# Sidebar for global controls
with st.sidebar:
    st.title("NLP Suite PRO")
    
    if st.button("🏠 Home"):
        st.session_state.app_stage = "welcome"
        st.rerun()

    suite_mode = st.radio(
        "Choose your suite:",
        ("Light (Standard Models)", "PRO (Corpus-Aware Models)"),
        index=0 if st.session_state.suite_mode == "Light" else 1,
        key="suite_mode_radio",
        on_change=lambda: st.session_state.update({"app_stage": "welcome", "pro_models_loaded": False})
    )
    
    st.session_state.suite_mode = "Light" if "Light" in suite_mode else "PRO"

# --- Main App Logic ---

if st.session_state.app_stage == "welcome":
    st.title("Welcome to the Ultimate NLP Application Suite!")
    st.write("Your advanced toolkit for text analysis, generation, and more, now with a cosmic interface.")
    
    if st.button("🚀 Get Started"):
        if st.session_state.suite_mode == "PRO":
            st.session_state.app_stage = "pro_corpus_upload"
        else:
            st.session_state.app_stage = "analysis"
        st.rerun()

elif st.session_state.app_stage == "pro_corpus_upload":
    st.header("Step 1: Upload Your Corpus")
    
    uploaded_file = st.file_uploader(
        "Upload a .txt, .csv, or image file",
        type=['txt', 'csv', 'png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/plain":
                st.session_state.corpus = uploaded_file.getvalue().decode("utf-8")
                st.session_state.app_stage = "pro_preprocessing"
                st.rerun()
            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                st.session_state.temp_df = df
                st.session_state.app_stage = "pro_csv_column"
                st.rerun()
            elif uploaded_file.type.startswith("image/"):
                img = Image.open(uploaded_file)
                st.session_state.corpus = pytesseract.image_to_string(img)
                st.session_state.app_stage = "pro_preprocessing"
                st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

elif st.session_state.app_stage == "pro_csv_column":
    st.header("Step 1a: Select CSV Column")
    df = st.session_state.temp_df
    column = st.selectbox("Which column contains the text?", df.columns)
    
    if st.button("Confirm Column"):
        st.session_state.corpus = ' '.join(df[column].dropna().astype(str).tolist())
        st.session_state.app_stage = "pro_preprocessing"
        st.rerun()

elif st.session_state.app_stage == "pro_preprocessing":
    st.header("Step 2: Text Preprocessing")
    
    with st.form("preprocessing_form"):
        st.write("Select the cleaning and preprocessing steps to apply:")
        
        cols = st.columns(2)
        options = {
            "Remove HTML/URLs": cols[0].checkbox("Remove HTML/URLs", True),
            "Expand Chat Slang": cols[0].checkbox("Expand Chat Slang", True),
            "Lowercase": cols[0].checkbox("Lowercase", True),
            "Handle Emojis": cols[1].radio("Handle Emojis", ["Convert to Text", "Remove", "Ignore"]),
            "Spelling Correction": cols[0].checkbox("Spelling Correction", True, disabled=not SPELLCHECK_AVAILABLE),
            "Remove Punctuation": cols[1].checkbox("Remove Punctuation", True),
            "Remove Stopwords": cols[1].checkbox("Remove Stopwords", True),
            "Lemmatize": cols[1].checkbox("Lemmatize", True),
        }
        if not SPELLCHECK_AVAILABLE:
            cols[0].warning("`pyspellchecker` not installed. Spelling correction disabled.")
        
        submitted = st.form_submit_button("✨ Apply & Continue")
        if submitted:
            st.session_state.app_stage = "pro_processing_inprogress"
            st.session_state.preprocessing_options = options
            st.rerun()


elif st.session_state.app_stage == "pro_processing_inprogress":
    st.header("Applying Preprocessing...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    def progress_callback(progress, text):
        progress_bar.progress(int(progress*100))
        
        elapsed = time.time() - start_time
        if progress > 0.01:
            eta_seconds = int((elapsed / progress) * (1 - progress))
            mins, secs = divmod(eta_seconds, 60)
            eta_str = f"{mins:02d}m {secs:02d}s"
            status_text.text(f"{text}\n\nEstimated Time Remaining: {eta_str}")
        else:
            status_text.text(text)
    
    st.session_state.processed_corpus = preprocess_corpus(st.session_state.corpus, st.session_state.preprocessing_options, progress_callback)
    
    st.success("✅ Preprocessing Complete!")
    if st.button("▶ Continue to Model Selection"):
        st.session_state.app_stage = "pro_model_selection"
        st.rerun()


elif st.session_state.app_stage == "pro_model_selection":
    st.header("Step 3: Select a Base Model")
    
    model_choice = st.radio(
        "Choose a model to fine-tune with your corpus.",
        ("distilgpt2 (Fast & Lightweight)", "gpt2 (Standard & Robust)")
    )
    st.session_state.base_model_name = model_choice.split(" ")[0]

    st.header("Step 4: Select Training Depth")
    epoch_choice = st.select_slider(
        "Choose how extensively to fine-tune:",
        options=["Express (3 epochs)", "Standard (10 epochs)", "Expert (20 epochs)"],
        value="Express (3 epochs)"
    )
    st.session_state.epochs = int(re.search(r'\((\d+)', epoch_choice).group(1))

    if st.button("💪 Start Training"):
        st.session_state.app_stage = "analysis" 
        st.rerun()

elif st.session_state.app_stage == "analysis":
    st.title("Text Analysis")
    
    # This state dictionary is still useful if you want to
    # store models after they are loaded once.
    if 'pipelines' not in st.session_state:
        st.session_state.pipelines = {}

    if st.session_state.suite_mode == "PRO":
        if not st.session_state.pro_models_loaded:
            st.header("Simulating Fine-Tuning...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_duration = 5 
            start_time = time.time()

            for i in range(101):
                elapsed = time.time() - start_time
                progress = i / 100.0
                
                eta_seconds = int((elapsed / progress) * (1 - progress)) if progress > 0 else total_duration
                mins, secs = divmod(eta_seconds, 60)
                eta_str = f"{mins:02d}m {secs:02d}s"
                
                status_text.text(f"Fine-tuning {st.session_state.base_model_name}... (ETA: {eta_str})")
                progress_bar.progress(i)
                time.sleep(total_duration / 100)

            st.session_state.fine_tuned_model_path = st.session_state.base_model_name
            st.session_state.pro_models_loaded = True
            st.success(f"PRO Mode Active: Fine-tuning simulation complete!")
            st.balloons()
            time.sleep(2)
        
        # Set the model name, but don't load it yet
        next_word_model = st.session_state.fine_tuned_model_path if st.session_state.fine_tuned_model_path else "distilgpt2"
    else:
        next_word_model = "distilgpt2"
        
    #
    # --- CHANGE: REMOVED MODEL LOADING BLOCK ---
    # We no longer load all models here.
    # They will be loaded on-demand inside the "Analyze" button logic.
    #
    
    st.header("Choose an action and enter your text")
    
    action = st.selectbox(
        "What would you like to do?",
        ("✍️ Predict Next Word", "😊 Analyze Sentiment", "🎭 Detect Emotion", "📄 Summarize Long Text", "☁️ Create Word Cloud")
    )
    
    text_input = st.text_area("Enter your text here:", height=100)

    if st.button("🔎 Analyze"):
        if not text_input:
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                
                # --- CHANGE: 'Predict Next Word' Logic ---
                if "Predict Next Word" in action:
                    st.subheader("Next Word Prediction")
                    
                    # Load model on-demand
                    next_word_pipe = load_nlp_pipeline(next_word_model, "text-generation")
                    
                    if next_word_pipe:
                        if st.session_state.suite_mode == "PRO" and st.session_state.processed_corpus:
                            context = get_corpus_context(st.session_state.processed_corpus)
                            prompt = f"{context} {text_input}"
                            st.info(f"Using context from your corpus to improve prediction...")
                        else:
                            prompt = text_input

                        st.markdown("---")
                        for i, length in enumerate([3, 5, 7], 1):
                            result = next_word_pipe(prompt, max_new_tokens=length, num_return_sequences=1)[0]['generated_text']
                            prediction = result[len(prompt):].strip()
                            st.markdown(f"**Probable Output {i} ({length} words):**")
                            st.success(prediction)

                # --- CHANGE: 'Analyze Sentiment' Logic ---
                elif "Analyze Sentiment" in action:
                    st.subheader("Sentiment Analysis")
                    
                    # Load model on-demand
                    sentiment_pipe = load_nlp_pipeline("distilbert-base-uncased-finetuned-sst-2-english", "sentiment-analysis")
                    
                    if sentiment_pipe:
                        result = sentiment_pipe(text_input)[0]
                        sentiment_emojis = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}
                        emoji_icon = sentiment_emojis.get(result['label'], "")
                        st.write(f"**Sentiment:** {result['label']} {emoji_icon} ({result['score']:.2f})")

                # --- CHANGE: 'Detect Emotion' Logic ---
                elif "Detect Emotion" in action:
                    st.subheader("Emotion Detection")
                    
                    # Load model on-demand
                    emotion_pipe = load_nlp_pipeline("j-hartmann/emotion-english-distilroberta-base", "text-classification")

                    if emotion_pipe:
                        result = emotion_pipe(text_input)[0]
                        emotion_emojis = {"joy": "😂", "sadness": "😢", "anger": "😠", "fear": "😨", "love": "❤️", "surprise": "😮"}
                        emoji_icon = emotion_emojis.get(result['label'], "")
                        st.write(f"**Emotion:** {result['label']} {emoji_icon} ({result['score']:.2f})")
                
                # --- CHANGE: 'Summarize Long Text' Logic ---
                elif "Summarize Long Text" in action:
                    if len(text_input.split()) <= 30:
                        st.warning("Summarization works best on texts longer than 30 words.")
                    else:
                        st.subheader("Summarization")
                        
                        # Load model on-demand
                        summarization_pipe = load_nlp_pipeline("sshleifer/distilbart-cnn-12-6", "summarization")
                        
                        if summarization_pipe:
                            result = summarization_pipe(text_input, min_length=30, do_sample=False)[0]['summary_text']
                            st.write(result)
                
                # --- NO CHANGE: 'Create Word Cloud' Logic ---
                elif "Create Word Cloud" in action:
                    st.subheader("Sentiment Word Cloud")
                    try:
                        wc = WordCloud(width=800, height=400, background_color=None, mode="RGBA", colormap="viridis").generate(text_input)
                        fig, ax = plt.subplots()
                        fig.patch.set_alpha(0)
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        
                        buf = BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
                        
                        st.download_button(
                            label="📥 Download Word Cloud",
                            data=buf.getvalue(),
                            file_name="word_cloud.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Could not generate word cloud: {e}")