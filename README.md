Transfer Learning-Based Classification of Poultry Diseases

This project uses transfer learning and deep learning techniques to classify poultry diseases into four categories: Salmonella, New Castle Disease, Coccidiosis, and Healthy. It aims to assist farmers and veterinarians by providing a quick and efficient mobile-based diagnosis tool.

🧠 Project Objective

To build an intelligent, image-based classification system for poultry diseases using Convolutional Neural Networks (CNN) and integrate the model into a mobile application to support rural and commercial poultry farming.

📁 Dataset

Custom dataset of poultry fecal or symptom images.

Classes: Salmonella, New Castle, Coccidiosis, and Healthy

Format: Directory-based with subfolders per class.

Dataset Path:

C:\Users\karth\Downloads\My First Project.v2i.yolokeras.zip

🧪 Model Architecture

Implemented using TensorFlow and Keras:

Conv2D Layers (32 and 64 filters)

MaxPooling2D

Dense layers with ReLU and Softmax

Dropout for regularization

🧬 Transfer Learning Alternative

Model can be improved with transfer learning using MobileNetV2 or ResNet50 for better accuracy and generalization.

📦 Requirements

pip install tensorflow numpy matplotlib

🚀 How to Run

Unzip the dataset and organize into train/ and valid/ folders with class subfolders.

Update paths in the Python script.

Train the model:

model.fit(train_data, validation_data=val_data, epochs=10)

Save model:

model.save("your_model_name.h5")

📱 Mobile App Integration

The final model can be integrated into a mobile app using:

TensorFlow Lite (TFLite) for Android

Flask API for web/mobile interaction

✅ Results

Model accuracy: ~95%

Tested on real symptom images

Effective classification even in low-resource settings

📌 Use Cases

🧑‍🌾 Rural farmer diagnosing outbreak

🐓 Commercial poultry farm early warning system

🧑‍⚕️ Veterinary student training tool

📚 License

MIT License

🔗 Links

Zenodo Dataset (Alternative)

Project Report DOCX

GitHub Repository

✨ Made with care to improve poultry health and farmer livelihoods.

