
#Hazar michael 1201838
#Rania Rimawi 1201179
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import Entry, Label, Button, Toplevel, Canvas
from PIL import Image, ImageTk


# Load and preprocess the dataset
file_path = 'ramallah 2022-08-01 to 2024-01-01.csv'
data = pd.read_csv(file_path)
data['datetime'] = pd.to_datetime(data['datetime'])
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data = data.drop('datetime', axis=1)

label_encoder = LabelEncoder()
data['icon'] = label_encoder.fit_transform(data['icon'])

X = data.drop('icon', axis=1)
y = data['icon']


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Train ANN Classifier
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)


# Testing the Decision Tree Classifier
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = dt_classifier.score(X_test, y_test)

# Transform predictions and test labels to original string format
dt_predictions_transformed = label_encoder.inverse_transform(dt_predictions)
y_test_transformed = label_encoder.inverse_transform(y_test)

# Generate classification report for Decision Tree using string labels
dt_report_df = pd.DataFrame(classification_report(y_test_transformed, dt_predictions_transformed,
                                                  output_dict=True)).transpose()
dt_report_df = dt_report_df.drop('support', axis=1)  # Remove the 'support' column

# Testing the ANN
ann_predictions = np.argmax(ann_model.predict(X_test_scaled), axis=1)
ann_accuracy = ann_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
ann_predictions_transformed = label_encoder.inverse_transform(ann_predictions)

# Generate classification report for ANN using string labels
ann_report_df = pd.DataFrame(classification_report(y_test_transformed, ann_predictions_transformed,
                                                   output_dict=True)).transpose()
ann_report_df = ann_report_df.drop('support', axis=1)  # Remove the 'support' column


# Prediction Function
def predict_weather(input_data):
    input_df = pd.DataFrame([input_data], columns=X_train.columns)  # Use DataFrame with columns
    input_data_scaled = scaler.transform(input_df)
    dt_pred = dt_classifier.predict(input_df)[0]
    ann_pred = np.argmax(ann_model.predict(input_data_scaled), axis=1)[0]
    return label_encoder.inverse_transform([dt_pred])[0], label_encoder.inverse_transform([ann_pred])[0]



def create_table_window(title, accuracy, report_df):
    top = Toplevel()
    top.title(title)
    top_width = 600
    top_height = 400
    screen_width = top.winfo_screenwidth()
    screen_height = top.winfo_screenheight()
    center_x = int(screen_width / 2 - top_width / 2)
    center_y = int(screen_height / 2 - top_height / 2)
    top.geometry(f'{top_width}x{top_height}+{center_x}+{center_y}')
    top.configure(bg='#D5E0F7')

    accuracy_label = Label(top, text=f"Accuracy: {accuracy:.4f}", font=("Helvetica", 16), bg='#D5E0F7')
    accuracy_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=10, pady=10)

    headers = ["", "precision", "recall", "f1-score"]
    for i, header in enumerate(headers):
        label = Label(top, text=header, font=("Helvetica", 14, "bold"), bg='#D5E0F7')
        label.grid(row=1, column=i, padx=10, pady=5)

    for i, (index, row) in enumerate(report_df.iterrows()):
        index_label = Label(top, text=index, font=("Helvetica", 12), bg='#D5E0F7')
        index_label.grid(row=i + 2, column=0, sticky="w", padx=10)
        for j, value in enumerate(row):
            value_label = Label(top, text=f"{value:.3f}" if isinstance(value, float) else value, font=("Helvetica", 12), bg='#D5E0F7')
            value_label.grid(row=i + 2, column=j + 1, padx=10)

def show_dt_results():
    create_table_window("Decision Tree Results", dt_accuracy, dt_report_df)

def show_ann_results():
    create_table_window("ANN Results", ann_accuracy, ann_report_df)

#function to show whose accuracy is higher between the two models
def show_comparison_results():
    accuracies = {
        'Decision Tree': dt_accuracy,
        'ANN': ann_accuracy,

    }
    # Sort by accuracy
    sorted_accuracies = dict(sorted(accuracies.items(), key=lambda item: item[1], reverse=True))

    # Create and show comparison window
    create_comparison_window("Model Accuracy Comparison", sorted_accuracies)

def create_comparison_window(title, accuracies):
    top = Toplevel()
    top.title(title)
    top_width = 600
    top_height = 400
    screen_width = top.winfo_screenwidth()
    screen_height = top.winfo_screenheight()
    center_x = int(screen_width / 2 - top_width / 2)
    center_y = int(screen_height / 2 - top_height / 2)
    top.geometry(f'{top_width}x{top_height}+{center_x}+{center_y}')
    top.configure(bg='#D5E0F7')

    # Finding the model with the highest accuracy
    highest_accuracy_model = max(accuracies, key=accuracies.get)
    highest_accuracy = accuracies[highest_accuracy_model]

    # Display accuracies in percentage format
    for model, accuracy in accuracies.items():
        accuracy_percentage = accuracy * 100  # Convert to percentage
        label = Label(top, text=f"{model} Accuracy: {accuracy_percentage:.3f}%", font=("Helvetica", 14), bg='#D5E0F7')
        label.pack(pady=5)

    # Display comparison sentence with percentage format
    highest_accuracy_percentage = highest_accuracy * 100
    comparison_sentence = f"The model with the highest accuracy is {highest_accuracy_model} ({highest_accuracy_percentage:.3f}%)."
    comparison_label = Label(top, text=comparison_sentence, font=("Helvetica", 14, "bold"), bg='#D5E0F7', fg="green")
    comparison_label.pack(pady=10)


def custom_messagebox(title, message):
    top = tk.Toplevel()
    top.title(title)
    top_width = 600
    top_height = 400
    screen_width = top.winfo_screenwidth()
    screen_height = top.winfo_screenheight()
    center_x = int(screen_width / 2 - top_width / 2)
    center_y = int(screen_height / 2 - top_height / 2)
    top.geometry(f'{top_width}x{top_height}+{center_x}+{center_y}')
    top.configure(bg='#D5E0F7')

    # Calculate and set the text label's width
    text_width = top_width - 40  # Adjust for padding

    # Create a Label widget to display the message
    message_label = tk.Label(top, text=message, wraplength=text_width, justify='left', font=("Helvetica", 12, "bold"),
                             bg='#D5E0F7')
    message_label.pack(pady=20)

    # Prevent interaction with the main window while the custom messagebox is open
    top.transient(root)
    top.grab_set()
    root.wait_window(top)


def get_input():
    try:
        input_data = [float(entry.get()) for entry in entries]
        dt_pred, ann_pred = predict_weather(input_data)
        custom_messagebox("Predictions", f"\n\n\n\nDecision Tree Prediction: \t{dt_pred}\n\n\nANN Prediction:\t {ann_pred}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values.")

def exit_application():
    root.destroy()

root = tk.Tk()
root.title("Weather Prediction")

bg_color = '#C6D6F5'
root.configure(bg=bg_color)

# Calculate the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the x and y coordinates for the center of the screen
x = (screen_width - 700) // 2
y = (screen_height - 700) // 2

# Set the window size and position it in the center
root.geometry("700x700+{}+{}".format(x, y))

# Load and display an image at the top of the window
image = Image.open("weather1.png")  # Replace with your image path
image = image.resize((700, 250), Image.LANCZOS)  # Resize to (width, height)
photo = ImageTk.PhotoImage(image)

image_label = tk.Label(root, image=photo, bg=bg_color)
image_label.image = photo  # Keep a reference to avoid garbage collection
image_label.grid(row=0, column=0, columnspan=2, pady=(10, 20))


labels = [
    'Max Temperature     (°C)',
    'Min Temperature     (°C)',
    'Average Temperature (°C)',
    'Dew Point           (°C)',
    'Humidity            (%)',
    'Wind Speed          (km/h)',
    'Wind Direction      (°)',
    'Cloud Cover         (%)',
    'Year                (YYYY)',
    'Month               (MM)',
    'Day                 (DD)'
]

entries = []

start_row_index = 1

for i, label_text in enumerate(labels):
    label = tk.Label(root, text=label_text, bg=bg_color)
    label.grid(row=start_row_index + i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=start_row_index + i, column=1)
    entries.append(entry)

button_frame = tk.Frame(root, bg=bg_color)
button_frame.grid(row=12, column=0, columnspan=2, pady=5)

# Create buttons and add them to the button_frame
buttons = [
    ("Show Decision Tree Evaluation results", show_dt_results),
    ("Show ANN Evaluation Results", show_ann_results),
    ("Compare Model Accuracies", show_comparison_results),
    ("Predict Weather", get_input),
    ("Exit", exit_application)

]

for text, command in buttons:
    button = tk.Button(button_frame, text=text, command=command, bg='white')
    button.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

root.mainloop()