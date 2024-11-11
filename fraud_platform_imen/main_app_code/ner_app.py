import cv2
import spacy
from spacy.util import filter_spans
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
from spacy import displacy
import easyocr
import streamlit as st
import torch
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize EasyOCR and NLTK stopwords
reader = easyocr.Reader(['en'])
stop_words = set(stopwords.words("english"))

# Load Spacy model
# nlp = spacy.load("my_saved_model")
nlp = spacy.load("en_core_web_md")
nlp = spacy.load("my_saved_model", config={"include_static_vectors": False})

#--------------------------------- Image Preprocessing -----------------------------------
def preprocess_image(opencv_image):
    """Preprocess image by converting to grayscale, applying Gaussian Blur, and adaptive thresholding."""
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    processed_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return processed_image

#--------------------------------- OCR Extraction -----------------------------------
def ocr_extraction(opencv_image):
    """Extracts text from an image using EasyOCR."""
    processed_image = preprocess_image(opencv_image)
    result = reader.readtext(processed_image, paragraph="False")
    
    text = ' '.join([res[1] for res in result])
    return text, result

def draw_contours(opencv_image, result):
    """Draws bounding boxes around detected text regions in the image."""
    for detection in result:
        top_left, bottom_right = tuple(detection[0][0]), tuple(detection[0][2])
        opencv_image = cv2.rectangle(opencv_image, top_left, bottom_right, (0, 255, 0), 3)
    return opencv_image

#--------------------------------- Named Entity Recognition -----------------------------------
def perform_named_entity_recognition(text):
    """Performs Named Entity Recognition (NER) on input text."""
    if isinstance(text, str):
        doc = nlp(text)
        return doc
    raise ValueError("Input to NER must be a string.")

def color_gen():
    """Generates a random hex color code."""
    return f'#{random.randint(0, 0xFFFFFF):06x}'

def display_doc(doc):
    """Renders NER entities in HTML format with random colors for Streamlit display."""
    colors = {ent.label_: color_gen() for ent in doc.ents}
    options = {"ents": [ent.label_ for ent in doc.ents], "colors": colors}
    html = displacy.render(doc, style='ent', options=options, page=True, minify=True)
    st.write(html, unsafe_allow_html=True)
    return html

#--------------------------------- Fraud Detection -----------------------------------
@st.cache_data
def ner_list_similarity_jaccard(ner_list1, ner_list2):
    """Calculates Jaccard similarity between two lists of NER tokens after removing stopwords."""
    filtered_ner_list1 = [token for token in ner_list1 if token.lower() not in stop_words]
    filtered_ner_list2 = [token for token in ner_list2 if token.lower() not in stop_words]
    
    intersection_size = len(set(filtered_ner_list1).intersection(filtered_ner_list2))
    union_size = len(set(filtered_ner_list1).union(filtered_ner_list2))
    
    return intersection_size / union_size if union_size else 0

def fraud(detail):
    """Identifies potential fraud by comparing extracted NER with database entries using Jaccard similarity."""
    if not isinstance(detail, spacy.tokens.Doc):
        raise ValueError("Input to `fraud` must be a SpaCy Doc object.")
    
    try:
        df = pd.read_csv(r"fraud_platform_imen\data\cleaned_medicine_data.csv")
    except FileNotFoundError:
        st.error("Medicine data file not found.")
        return None, None, "Data file missing"
    
    # Data preparation
    df.fillna(' ', inplace=True)
    
    # Extract entities from SpaCy Doc object
    entity_dict = {label: [] for label in ["sub_category", "product_name", "salt_comp", "manufactured_by"]}
    for ent in detail.ents:
        if ent.label_ in entity_dict:
            entity_dict[ent.label_].append(ent.text)
    
    flattened_list = [item for sublist in entity_dict.values() for item in sublist]
    example_tokens = word_tokenize(' '.join(flattened_list))
    
    max_jaccard_score, max_jaccard_index = -1, -1
    
    # Calculate Jaccard similarity for each row in the DataFrame
    for index, row in df.iterrows():
        base_ner_list = [row[col] for col in ["product_name", "manufactured_by", "salt_comp", "sub_category"]]
        base_tokens = word_tokenize(' '.join(base_ner_list))
        
        # Filter tokens
        filtered_base_tokens = [token for token in base_tokens if token.lower() not in stop_words and token not in [',', '.', ':', 'nan']]
        filtered_example_tokens = [token for token in example_tokens if token.lower() not in stop_words and token not in [',', '.', ':', 'nan']]
        
        # Calculate Jaccard similarity
        jaccard_similarity = ner_list_similarity_jaccard(filtered_base_tokens, filtered_example_tokens)
        
        if jaccard_similarity > max_jaccard_score:
            max_jaccard_score, max_jaccard_index = jaccard_similarity, index

    # Retrieve entity data and set fraud status based on threshold
    threshold = 0.1
    entities = df.loc[max_jaccard_index] if max_jaccard_index != -1 else None
    fraud_status = "This Drug is potentially safe" if max_jaccard_score > threshold else "This Drug is potentially fraudulent"
    
    return max_jaccard_score, entities, fraud_status
