from flask import Flask, render_template, request
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

app = Flask(__name__)

# Define dataset paths
test_dataset_path = r"C:\Users\bandi\OneDrive\Documents\Bone Classification (2)\Bone Classification\test"
train_dataset_path = r"C:\Users\bandi\OneDrive\Documents\Bone Classification (2)\Bone Classification\train"

# Define image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# Add an output layer with softmax activation for 4 classes
predictions = Dense(4, activation='softmax')(x)

# Model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pre-trained layers except the last few
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train or load pre-trained model
if not os.path.exists('trained_model.h5'):
    # Create data generators with aggressive data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # Train model
    model.fit(train_generator,
              epochs=20,
              validation_data=test_generator)

    # Save trained model
    model.save('trained_model.h5')
else:
    # Load pre-trained model
    model = load_model('trained_model.h5')

    # Define data generators for test dataset
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        test_dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        img_path = "uploads/" + file.filename
        file.save(img_path)
        predicted_class = predict_class(img_path)
        return render_template('index.html', pred=predicted_class, img_path=img_path)
    except Exception as e:
        print(e)
        return render_template('index.html', error='Error processing file')

def predict_class(img_path):
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import preprocess_input
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))
    prediction = model.predict(img_array)
    predicted_class = test_generator.class_indices
    for key, value in predicted_class.items():
        if value == np.argmax(prediction):
            return key

# Run the Flask application
if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
