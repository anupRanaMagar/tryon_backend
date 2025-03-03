import tensorflow as tf
import numpy as np
from PIL import Image

model_path = 'model/cloth_mask_model.h5'

# Load model
model = tf.keras.models.load_model(model_path)

def preprocess_image(img, img_size=128):
    img = img.resize((img_size, img_size))  # Resize using PIL
    img = np.array(img) / 255.0  # Convert to NumPy array and normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

def predict_cloth_mask(image, model, img_size=128):
    image = preprocess_image(image, img_size)
    prediction = model.predict(image)
    prediction = prediction[0, :, :, 0]  # Remove batch and channel dimensions
    return prediction

def predic(cloth):
    # Load image using PIL
    cloth = cloth.convert("RGB")

    # Get the mask prediction
    output = predict_cloth_mask(cloth, model)

    output = (output * 255).astype(np.uint8)  # Scale back to [0, 255]

    # Resize to 768x1024 using PIL
    output_resized = Image.fromarray(output).resize((768, 1024), Image.NEAREST)

    return output_resized


