import streamlit as st
import torch
from PIL import Image
import torch.nn as nn
from resnet import ResNet, ResidualBlock, transform

# Set page config
st.set_page_config(
    page_title="Weather Image Classifier",
    page_icon="🌤️"
)

# Define the classes
CLASSES = {
    0: 'dew',
    1: 'fogsmog',
    2: 'frost',
    3: 'glaze',
    4: 'hail',
    5: 'lightning',
    6: 'rain',
    7: 'rainbow',
    8: 'rime',
    9: 'sandstorm',
    10: 'snow'
}


@st.cache_resource
def load_model():
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet(
        residual_block=ResidualBlock,
        n_blocks_lst=[2, 2, 2, 2],
        n_classes=len(CLASSES)
    ).to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(
        'week-2\\checkpoints\\resnet.pt', map_location=device, weights_only=True))
    model.eval()
    return model, device


def predict_image(image, model, device):
    # Transform the image
    img_tensor = transform(image)

    # Add batch dimension and move to device
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return CLASSES[predicted.item()]


def main():
    st.title("Weather Image Classifier 🌤️")
    st.write("Upload an image to classify the weather condition!")

    # Load model
    try:
        model, device = load_model()
    except Exception as e:
        st.error("Error loading the model. Make sure the model weights file exists.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Add a predict button
        if st.button('Predict'):
            with st.spinner('Analyzing image...'):
                try:
                    # Get prediction
                    prediction = predict_image(image, model, device)

                    # Display result
                    st.success(f"Prediction: {prediction.upper()}")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    main()
