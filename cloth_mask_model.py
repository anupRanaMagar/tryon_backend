import tensorflow as tf
import tensorflow as tf
import cv2
import numpy as np

model_path = 'model/cloth_mask_model.h5'
# cloth_path = 'datasets/test/cloth/08348_00.jpg'
output_path = 'output_mask.jpg'  # Output file path

# Load model
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path, img_size=128):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

def predict_cloth_mask(image_path, model, img_size=128):
    image = preprocess_image(image_path, img_size)
    prediction = model.predict(image)
    # predection = cv2.resize(prediction, (768, 1024))    
    prediction = prediction[0, :, :, 0]  # Remove batch and channel dimensions
    return prediction


def predic(cloth_path):
    # Get the mask prediction
    output = predict_cloth_mask(cloth_path, model)

    output = (output * 255).astype(np.uint8)  # Scale back to [0, 255]

    # Resize to 768x1024
    output_resized = cv2.resize(output, (768, 1024), interpolation=cv2.INTER_NEAREST)

    # Save as a JPEG file
    cv2.imwrite(output_path, output_resized)

