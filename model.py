import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and preprocess image data (this assumes you have images labeled)
def load_image_data(image_folder_path):
    # Load and preprocess images into numpy arrays
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_data = datagen.flow_from_directory(
        image_folder_path,
        target_size=(256, 256),  # Resize images to a uniform size
        batch_size=32,
        class_mode='categorical',  # Assuming multiple classes or patterns
        subset='training'
    )
    
    validation_data = datagen.flow_from_directory(
        image_folder_path,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_data, validation_data

# Define the CNN model
def create_dendrite_model(input_shape):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(3, activation='softmax'))  # Assuming 3 classes for patterns

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Train the model
def train_dendrite_model(model, train_data, validation_data, epochs=20):
    history = model.fit(train_data, validation_data=validation_data, epochs=epochs)
    
    return history

# Visualize model performance
def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

# Main function
if __name__ == '__main__':
    # Load your image dataset path here
    image_folder_path = 'path_to_dendrite_images'
    
    train_data, validation_data = load_image_data(image_folder_path)
    
    # Create the model
    input_shape = (256, 256, 3)  # Assuming RGB images
    model = create_dendrite_model(input_shape)
    
    # Train the model
    history = train_dendrite_model(model, train_data, validation_data, epochs=20)
    
    # Plot the training history
    plot_training_history(history)
    
    # Save the model
    model.save('dendrite_growth_prediction_model.h5')

    # To predict on new data
    # new_image = load_and_preprocess('path_to_new_image')
    # prediction = model.predict(new_image)
