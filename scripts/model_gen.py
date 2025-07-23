import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

# 1. Load or create dataset
data = {
    "Temperature": [35, 22, 18, 28, 15, 32, 21, 25],
    "Humidity":    [30, 85, 90, 55, 95, 40, 80, 65],
    "Condition":   ["Hot", "Rainy", "Foggy", "Sunny", "Foggy", "Hot", "Rainy", "Sunny"]
}

df = pd.DataFrame(data)

# 2. Preprocess
X = df[["Temperature", "Humidity"]].values
y = df["Condition"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("*** Mean:", scaler.mean_)
print("*** Std:", scaler.scale_)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2)

# 3. Build and train model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc*100:.2f}%")

# Save model
model.save("weather_model.h5")

# Add at the bottom of the same file or as a separate script
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional for quantization
tflite_model = converter.convert()

with open("weather_model.tflite", "wb") as f:
    f.write(tflite_model)

