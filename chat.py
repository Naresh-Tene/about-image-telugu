import streamlit as st
import requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Function to load the captioning model (runs only once) ---
@st.cache_resource
def load_captioning_model():
    """Loads the BLIP image captioning model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

# --- Function to call the Translation API ---
def translate_to_telugu(api_key, text_to_translate):
    """Calls the Hugging Face API to translate English text to Telugu."""
    api_url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-te"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": text_to_translate}
    
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        # Provide a more detailed error if translation fails
        return {"error": f"అనువాదం విఫలమైంది. లోపం: {response.text}"}

# --- Streamlit App UI in Telugu ---
st.set_page_config(layout="wide", page_title="చిత్ర వివరణ జనరేటర్")
st.title("🖼️ చిత్ర వివరణ జనరేటర్")
st.write("చిత్రాన్ని అప్‌లోడ్ చేయండి మరియు ఈ అనువర్తనం దాని కోసం వివరణను రూపొందిస్తుంది.")

# --- Get API Key from Streamlit Secrets (needed for translation) ---
try:
    hf_api_key = st.secrets["HF_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("అనువాదం కోసం Hugging Face API కీ మీ Streamlit రహస్యాలలో కనుగొనబడలేదు.")
    st.stop()

# --- Load the local model ---
with st.spinner("మోడల్‌ను లోడ్ చేస్తోంది... దీనికి కొంత సమయం పట్టవచ్చు."):
    caption_processor, caption_model = load_captioning_model()

# --- Main App Logic ---
uploaded_file = st.file_uploader("ఇక్కడ ఒక చిత్రాన్ని ఎంచుకోండి...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="అప్‌లోడ్ చేసిన చిత్రం")

    with col2:
        with st.spinner("వివరణను రూపొందిస్తోంది..."):
            # Generate English caption locally
            inputs = caption_processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values
            output_ids = caption_model.generate(pixel_values, max_length=50, num_beams=4)
            english_caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)

            # Translate the caption to Telugu using the API
            translation_result = translate_to_telugu(hf_api_key, english_caption)
            
            if "error" in translation_result:
                st.error(translation_result['error'])
            else:
                telugu_caption = translation_result[0]['translation_text']
                st.success(f"**తెలుగు వివరణ:**\n\n{telugu_caption}")
