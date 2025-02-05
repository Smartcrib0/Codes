# Importing required libraries for data processing, modeling, and visualization
import os  # For file and directory operations
import librosa  # For audio processing
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from keras.models import Model  # For creating the CNN model
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, 
                          Dropout, BatchNormalization)  # Layers for CNN
from keras.optimizers import Adam  # Optimizer for training
from keras.utils import to_categorical  # To convert labels into one-hot encoding
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Callbacks for better training
import tensorflow as tf  # For using GPU and deep learning utilities

# Check if GPUs are available for faster computation
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

# Dataset path and the audio classes to classify
data_dir = 'E:/Senior_Project/Sound_Detection/aug-seniorDatasetX'  # Path to the dataset
classes = ['belly_pain', 'burping', 'cold-hot', 'discomfort', "dontKnow", 
           'hungry', 'lonely', 'scared', 'tired']  # Cry categories
target_length = 6 * 22050  # Define 6 seconds of audio with a sample rate of 22050 Hz

# Function to process a single audio file
def process_audio(file_path, duration_seconds=6, target_sr=22050):
    """Load an audio file and extract mel-spectrogram features."""
    try:
        y, sr = librosa.load(file_path, sr=target_sr)  # Load the audio file
        y = librosa.util.fix_length(y, size=target_sr * duration_seconds)  # Ensure fixed length
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # Compute mel-spectrogram
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibel scale
        return mel_spec_db  # Return mel-spectrogram as features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")  # Handle errors gracefully
        return None  # Return None if processing fails

# Function to process the entire dataset
def process_sounds(data_dir, classes, duration_seconds=6):
    """Extract features and labels from all audio files."""
    data = []  # Store features
    labels = []  # Store labels
    for i, class_name in enumerate(classes):  # Iterate through each class
        class_dir = os.path.join(data_dir, class_name)  # Path to class folder
        for filename in os.listdir(class_dir):  # Iterate through files in the folder
            if filename.endswith('.wav'):  # Process only .wav files
                file_path = os.path.join(class_dir, filename)
                mel_spec_db = process_audio(file_path, duration_seconds=duration_seconds)  # Process file
                if mel_spec_db is not None:  # If processing is successful
                    data.append(mel_spec_db)  # Add features to data
                    labels.append(i)  # Add label as class index
    return np.array(data), np.array(labels)  # Return features and labels as arrays

# Function to define and build the CNN model
def build_model(input_shape, num_classes):
    """Create a CNN model for sound classification."""
    inputs = Input(shape=input_shape)  # Input layer with given shape
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)  # 1st Conv layer
    x = BatchNormalization()(x)  # Normalize activations
    x = MaxPooling2D((2, 2))(x)  # Reduce spatial dimensions
    x = Dropout(0.25)(x)  # Dropout to prevent overfitting

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # 2nd Conv layer
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)  # 3rd Conv layer
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)  # Flatten to connect to dense layers
    x = Dense(512, activation='relu')(x)  # Dense layer with 512 neurons
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # Output layer with softmax activation

    model = Model(inputs, outputs)  # Define the model
    model.compile(optimizer=Adam(learning_rate=0.0001),  # Compile with Adam optimizer
                  loss='categorical_crossentropy',  # Use categorical cross-entropy loss
                  metrics=['accuracy'])  # Track accuracy during training
    return model  # Return the built model

# Main function to run the process
def main():
    # Step 1: Process the dataset
    X, y = process_sounds(data_dir, classes, duration_seconds=6)  # Extract features and labels

    # Normalize features and add a channel dimension for CNN
    X = X.astype('float32') / np.max(X)  # Normalize to [0, 1]
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    y = to_categorical(y, num_classes=len(classes))  # Convert labels to one-hot encoding

    # Step 2: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Build the CNN model
    input_shape = X_train.shape[1:]  # Determine input shape for the model
    model = build_model(input_shape, num_classes=len(classes))

    # Define callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32,
                        callbacks=[early_stopping, lr_scheduler])

    # Step 4: Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Step 5: Save the trained model
    model.save('sound_detection_modelCNN.h5')  # Save the model to a file
    print("Model saved as 'sound_detection_modelCNN.h5'.")
    
# Run the main function
if __name__ == "__main__":
    main()
