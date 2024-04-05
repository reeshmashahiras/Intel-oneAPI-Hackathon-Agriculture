import streamlit as st
import joblib
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models

# Set page configurations
st.set_page_config(
    page_title="🌱 Easy Farming",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="auto",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(to bottom, #00843D, #00CBA4);
        color: white;
    }
    .css-1i3mj7p {
        background-color: #00CBA4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom styling for title
st.title(
    "Easy Farming",
    anchor="center",
)
st.subheader("🌿 Your Digital Farming Companion")

# Sidebar
with st.sidebar:
    st.image("farm_icon (1).png", width=100)
    st.markdown("## 🌱 MENU  🌱 ")
    selected_page = st.selectbox(
        "",
        ("Home", "Crop Recommendation","Plant Disease Detection"),
    )

if selected_page == "Home":
    st.markdown("## Welcome to Easy Farming!")
    st.write(
        "Step into the realm of intelligent agriculture. Start your exploration by choosing a feature from the sidebar."

    )
    st.image("farm_image (1).jpg", width=500)

elif selected_page == "Crop Recommendation":
    crop_names = {
    0: 'rice',
    1: 'maize',
    2: 'chickpea',
    3: 'kidneybeans',
    4: 'pigeonpeas',
    5: 'mothbeans',
    6: 'mungbean',
    7: 'blackgram',
    8: 'lentil',
    9: 'pomegranate',
    10: 'banana',
    11: 'mango',
    12: 'grapes',
    13: 'watermelon',
    14: 'muskmelon',
    15: 'apple',
    16: 'orange',
    17: 'papaya',
    18: 'coconut',
    19: 'cotton',
    20: 'jute',
    21: 'coffee'
}

    st.markdown("## 🌾 Crop Recommendation System")
    st.write(
        "Need guidance on choosing the right crop? We've got you covered. Provide some soil and weather information, and let us recommend the best crop for you!"
    )
    st.header("Enter Values for Crop Recommendation")
    
    nitrogen = st.number_input("Nitrogen in Soil (kg/ha)")
    phosphorus = st.number_input("Phosphorous in Soil (kg/ha)")
    potassium = st.number_input("Potassium in Soil (kg/ha)")
    temperature = st.number_input("Average Temperature (°C)")
    humidity = st.number_input("Average Humidity (%)")
    ph_value = st.number_input("Average pH Value")
    rainfall = st.number_input("Rainfall Amount (mm)")
    

    # Load the trained model
    model = joblib.load('Random Forest_model.pkl')

    if st.button("Predict Crop"):
        # Convert input to a format suitable for prediction
        user_input_array = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])
        # Predict the crop using the trained model
        predicted_crop_class = model.predict(user_input_array)[0]
        # Map predicted crop class to crop name
        predicted_crop_name = crop_names.get(predicted_crop_class, "Unknown")
        st.success(f"Predicted Crop: **{predicted_crop_name}**")

if selected_page == "Plant Disease Detection":
    # Define class_index_to_name dictionary
    class_index_to_name = {
    1: 'Apple___Apple_scab', 
    2: 'Apple___Black_rot', 
    3: 'Apple___Cedar_apple_rust', 
    4: 'Apple___healthy',
    5: 'Blueberry___healthy', 
    6: 'Cherry_(including_sour)___Powdery_mildew', 
    7: 'Cherry_(including_sour)___healthy',
    8: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    9: 'Corn_(maize)___Common_rust_', 
    10: 'Corn_(maize)___Northern_Leaf_Blight', 
    11: 'Corn_(maize)___healthy', 
    12: 'Grape___Black_rot', 
    13: 'Grape___Esca_(Black_Measles)', 
    14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    15: 'Grape___healthy',
    16: 'Orange___Haunglongbing_(Citrus_greening)', 
    17: 'Peach___Bacterial_spot', 
    18: 'Peach___healthy',
    19: 'Pepper,_bell___Bacterial_spot', 
    20: 'Pepper,_bell___healthy', 
    21: 'Potato___Early_blight',
    22: 'Potato___Late_blight', 
    23: 'Potato___healthy', 
    24: 'Raspberry___healthy', 
    25: 'Soybean___healthy',
    26: 'Squash___Powdery_mildew', 
    27: 'Strawberry___Leaf_scorch', 
    28: 'Strawberry___healthy',
    29: 'Tomato___Bacterial_spot', 
    30: 'Tomato___Early_blight', 
    31: 'Tomato___Late_blight',
    32: 'Tomato___Leaf_Mold', 
    33: 'Tomato___Septoria_leaf_spot', 
    34: 'Tomato___Spider_mites Two-spotted_spider_mite',
    35: 'Tomato___Target_Spot', 
    36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    37: 'Tomato___Tomato_mosaic_virus',
    38: 'Tomato___healthy'
}

    class CnnModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            resnet = models.resnet18(pretrained=True)
            self.resnet_layers = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.fc1 = torch.nn.Linear(512, 128)
            self.fc2 = torch.nn.Linear(128, 38)
        
        def forward(self, xb):
            x = self.resnet_layers(xb)
            x = x.view(x.size(0), -1)
            out = torch.relu(self.fc1(x))
            out = self.fc2(out)
            return out

    @st.cache_data
    def load_model():
        model = CnnModel()
        model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model

    @st.cache_data
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        image = transform(image).unsqueeze(0)
        return image

    def main():
        st.title("Plant Disease Detection")

    # Load model
        model = load_model()

    # Upload image
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
            input_tensor = preprocess_image(image)

        # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = torch.argmax(output).item()

        # Display result
            predicted_label = class_index_to_name.get(predicted_class, "Unknown")
            st.success(f"Predicted Disease: {predicted_label}")

    if __name__ == "__main__":
        main()



