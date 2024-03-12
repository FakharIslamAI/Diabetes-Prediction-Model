import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
loaded_model = tf.keras.models.load_model("diabetes_prediction_model.h5")
scaler = StandardScaler()

# Function to predict diabetes
def predict_diabetes():
    # Prepare input data for prediction
    new_data = [float(entry.get()) for entry in entry_widgets]
    new_data = np.array(new_data).reshape(1, -1)
    new_data = scaler.transform(new_data)

    # Make prediction
    prediction = loaded_model.predict(new_data)

    # Display the result
    result_label.config(text=f"Diabetes Prediction: {prediction[0][0]}")

# Create GUI
root = tk.Tk()
root.title("Diabetes Prediction")

# Create entry widgets for input features
entry_widgets = []
feature_names = X.columns
for feature in feature_names:
    label = tk.Label(root, text=feature)
    label.pack()
    entry = tk.Entry(root)
    entry.pack()
    entry_widgets.append(entry)

# Create button for prediction
predict_button = tk.Button(root, text="Predict", command=predict_diabetes)
predict_button.pack()

# Display the prediction result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the GUI
root.mainloop()
