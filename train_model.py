from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'C:/Users/sjdin/OneDrive/Documents/deepfake_detection/train_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'C:/Users/sjdin/OneDrive/Documents/deepfake_detection/validation_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Print class indices to verify data loading
print("Class indices:", train_generator.class_indices)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
try:
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )
except Exception as e:
    print(f"An error occurred during training: {e}")

# Save the model

model.save('C:/Users/sjdin/OneDrive/Documents/deepfake_detection/deepfake_model.h5')
