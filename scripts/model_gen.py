import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

# ========= 1. DATASET GENERATION =========
data = {"Temperature": [], "Humidity": [], "Condition": []}

conditions = ["Hot", "Rainy", "Foggy", "Sunny"]
temperature_ranges = {"Hot": (32, 40), "Rainy": (20, 26), "Foggy": (14, 19), "Sunny": (25, 32)}
humidity_ranges = {"Hot": (30, 45), "Rainy": (82, 92), "Foggy": (90, 97), "Sunny": (50, 60)}

for cond in conditions:
    t_min, t_max = temperature_ranges[cond]
    h_min, h_max = humidity_ranges[cond]

    for temp in range(t_min, t_max + 1):       # loop temperature range
        for hum in range(h_min, h_max + 1):   # loop humidity range
            data["Temperature"].append(temp)
            data["Humidity"].append(hum)
            data["Condition"].append(cond)

df = pd.DataFrame(data)
print(df)
print("Total dataset size:", len(df))

# ========= 2. PREPROCESS =========
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

# ========= 3. TRAIN MODEL =========
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=500, validation_split=0.2)  # 40 epochs is enough now

# EVALUATE
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc*100:.2f}%")

# ========= 4. SAVE MODEL =========
model.save("weather_model.h5")

# ========= 5. EXPORT TFLITE =========
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # enables quantized model if possible
tflite_model = converter.convert()

with open("weather_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved successfully!")

# ========= 6. Make a sample prediction =========
X_new = np.array([[23.0, 36.0]])  # Example input
X_new_scaled = scaler.transform(X_new)
pred = model.predict(X_new_scaled)

print("Predicted condition:", label_encoder.inverse_transform([pred.argmax()])[0])

