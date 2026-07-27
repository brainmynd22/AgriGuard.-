import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

st.set_page_config(page_title="AgriGuard - Plant Disease Surveillance", page_icon="🌱", layout="wide")

st.title("🌱 AgriGuard: Intelligent Plant Disease Surveillance")
st.markdown("Upload an image of a **Cassava** or **Maize** leaf to analyze plant health in real time.")

# Sidebar controls
st.sidebar.header("Settings")
selected_crop = st.sidebar.selectbox("Select Crop", ["Cassava", "Maize"])
language = st.sidebar.selectbox("Language", ["English", "Yoruba", "Hausa", "Igbo"])

CROP_CLASSES = {
    "Cassava": ['Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 'Cassava healthy', 'Cassava mosaic'],
    "Maize": ['Maize fall armyworm', 'Maize grasshoper', 'Maize healthy', 'Maize leaf beetle', 'Maize leaf blight', 'Maize leaf spot']
}

MODEL_PATHS = {
    "Cassava": "saved_models/cassava_resnet50.pth",
    "Maize": "saved_models/maize_resnet50.pth"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(crop_name):
    num_classes = len(CROP_CLASSES[crop_name])
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    
    weights_path = MODEL_PATHS[crop_name]
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        st.error(f"Model file not found at {weights_path}. Please upload weights to saved_models/ folder.")
        return None

uploaded_file = st.file_uploader(f"Choose a {selected_crop} leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    model = load_model(selected_crop)
    
    if model and st.button("Analyze Leaf Health"):
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probs, 0)
            
        classes = CROP_CLASSES[selected_crop]
        detected = classes[predicted_idx.item()]
        score = confidence.item() * 100
        
        st.success(f"**Diagnosis:** {detected}")
        st.info(f"**Confidence Score:** {score:.2f}%")

