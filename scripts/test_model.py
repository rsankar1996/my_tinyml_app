import numpy as np
import tensorflow as tf  # or tflite_runtime.interpreter if using lite-only install
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(["Hot", "Rainy", "Foggy", "Sunny"])

print("Label classes:", le.classes_)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="weather_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare a test input (example: temperature=30°C, humidity=60%, pressure=1012 hPa)
test_input = np.array([[18.0, 60.0]], dtype=np.float32)
normalized_input = (test_input - [24.5, 67.5]) / [6.42261629, 22.5]

# If you used StandardScaler during training, normalize test_input here the same way

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], normalized_input.astype(np.float32))

# Run inference
interpreter.invoke()

# Get the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Raw output:", output_data)

# Get predicted label
predicted_index = np.argmax(output_data[0])
# labels = ["Hot", "Rainy", "Foggy", "Sunny", ]  # Use the same order as in your training

print("Predicted condition:", le.classes_[predicted_index])

