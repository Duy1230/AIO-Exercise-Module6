import streamlit as st
import torch
from torchvision import transforms
from lenet import LeNetClassifier
from PIL import Image
import io
import torch.nn.functional as F
import os
import random


def load_random_cassava_leaf_image():
    # Get list of image paths
    test_files = [os.path.join("./test_samples", file)
                  for file in os.listdir("./test_samples")]
    random_image = Image.open(random.choice(test_files))
    return random_image


# Load your model
class_names = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']
model = LeNetClassifier(num_classes=5)
model.load_state_dict(torch.load(
    'lenet_model.pt', map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

# Streamlit interface
st.title("Cassava Leaf Disease Classifier")
st.write("Upload an image of a cassava leaf and the model will predict what it is.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
random_image = None

with st.expander("Or use a random cassava leaf image"):
    if st.button("Generate"):
        random_image = load_random_cassava_leaf_image()


if uploaded_file or random_image is not None:
    # Read the image
    if uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = random_image
    # Adjust the width as needed
    st.image(image, caption='Uploaded Image.', width=500)

    # Preprocess the image
    image = transforms(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        confident_score = F.softmax(output, dim=1)[
            0][predicted]  # Get the confidence score

    # Display the prediction and confidence score
    st.write(f"Predicted Disease: {class_names[predicted.item()]}")
    # Display the confidence score
    st.write(f"Confidence Score: {confident_score.item():.4f}")
