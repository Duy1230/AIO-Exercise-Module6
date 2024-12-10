import streamlit as st
import torch
from torchvision import transforms
from lenet import LeNetClassifier, load_random_mnist_image
from PIL import Image
import io
import torch.nn.functional as F

# Load your model
model = LeNetClassifier(num_classes=10)
model.load_state_dict(torch.load('./model/lenet_model.pt'))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit interface
st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit and the model will predict what it is.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
random_image = None

with st.expander("Or use a random MNIST image"):
    if st.button("Generate"):
        random_image, random_label = load_random_mnist_image()


if uploaded_file or random_image is not None:
    # Read the image
    if uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = random_image
    # Adjust the width as needed
    st.image(image, caption='Uploaded Image.', width=100)

    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        confident_score = F.softmax(output, dim=1)[
            0][predicted]  # Get the confidence score

    # Display the prediction and confidence score
    st.write(f"Predicted Digit: {predicted.item()}")
    # Display the confidence score
    st.write(f"Confidence Score: {confident_score.item():.4f}")
