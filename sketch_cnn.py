# %% [markdown]
# # Sketch Recognition using CNN and Quick Draw Dataset

# %% [markdown]
# ### Package Install

# %%
# ! pip install tensorflow-gpu numpy matplotlib scikit-learn

# %%
import tensorflow as tf
import numpy as np
import os
import requests
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

# %%
# Enable mixed precision
# mixed_precision.set_global_policy('mixed_float16')

# %% [markdown]
# ### Download Data

# %%
BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

def download_data(category, output_dir):
    """Download Quick Draw dataset for a given category."""
    url = f"{BASE_URL}{category}.npy"
    output_path = os.path.join(output_dir, f"{category}.npy")
    
    if not os.path.exists(output_path):
        print(f"Downloading {category} dataset...")
        response = requests.get(url)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {category} dataset.")
    else:
        print(f"{category} dataset already exists.")

# %%
# Downloading Data for some categories

# Define categories and URLs
# CATEGORIES = ['cat', 'dog', 'car', 'airplane', 'apple', 'stairs']
CATEGORIES = [
    "aircraft carrier", "airplane", "alarm clock", "ambulance", "angel", 
    "animal migration", "ant", "anvil", "apple", "arm", "asparagus", "axe", 
    "backpack", "banana", "bandage", "barn", "baseball", "baseball bat", 
    "basket", "basketball", "bat", "bathtub", "beach", "bear", "beard", "bed", 
    "bee", "belt", "bench", "bicycle", "binoculars", "bird", "birthday cake", 
    "blackberry", "blueberry", "book", "boomerang", "bottlecap", "bowtie", 
    "bracelet", "brain", "bread", "bridge", "broccoli", "broom", "bucket", 
    "bulldozer", "bus", "bush", "butterfly", "cactus", "cake", "calculator", 
    "calendar", "camel", "camera", "camouflage", "campfire", "candle", "cannon", 
    "canoe", "car", "carrot", "castle", "cat", "ceiling fan", "cello", 
    "cell phone", "chair", "chandelier", "church", "circle", "clarinet", 
    "clock", "cloud", "coffee cup", "compass", "computer", "cookie", "cooler"
]

DATA_DIR = 'quickdraw_data'

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Download data for each category
for category in CATEGORIES:
    download_data(category, DATA_DIR)

# %% [markdown]
# ### DataLoader

# %%
class QuickDrawDataGenerator(Sequence):
    def __init__(self, data_dir, categories, samples_per_class, batch_size, subset='train'):
        self.data_dir = data_dir
        self.categories = categories
        self.samples_per_class = samples_per_class
        self.batch_size = batch_size
        self.subset = subset

        self.num_classes = len(categories)


    def __len__(self):
        return (self.samples_per_class * self.num_classes) // self.batch_size

    def __getitem__(self, idx):

        multipler = self.batch_size // len(self.categories)

        #print(f"${idx}")
        batch_x = np.empty((self.batch_size, 28, 28, 1), dtype=np.float32)
        #print(batch_x.shape)
        batch_labels = np.empty((self.batch_size), dtype=np.float32)
        for cat_idx, category in enumerate(self.categories):
            file_path = os.path.join(self.data_dir, f"{category}.npy")
            class_data = np.load(file_path)
            class_len = len(class_data)
            divider = (int)(class_len * 0.8)
            if self.subset == "train":
                class_data = class_data[0:divider]
            else:
                class_data = class_data[divider:]

            range_start = idx*multipler
            range_end = (idx+1)*multipler 
            if len(class_data) < range_end :
                range_start = 0
                range_end = multipler
            
            sample = class_data[range_start:range_end]
            sample = sample.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            #print(cat_idx)
            #print((cat_idx+1)*1000)
            #print(batch_x[cat_idx*1000:(cat_idx+1)*1000].shape)

            batch_x[cat_idx*multipler:(cat_idx+1)*multipler] = sample 
            batch_labels[cat_idx*multipler:(cat_idx+1)*multipler] = cat_idx

        
        # Convert to bfloat16
        # batch_x = tf.cast(batch_x, tf.bfloat16)
        return batch_x, tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)


        #batch_file_paths = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        #batch_indices = self.sample_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        #batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        #batch_x = np.empty((self.batch_size, 28, 28, 1), dtype=np.float32)
        #for i, (file_path, sample_idx) in enumerate(zip(batch_file_paths, batch_indices)):
        #    class_data = np.load(file_path)
        #    sample = class_data[sample_idx].reshape(28, 28, 1).astype('float32') / 255.0
        #    batch_x[i] = sample
        
        # Convert to bfloat16
        #batch_x = tf.cast(batch_x, tf.bfloat16)
        #return batch_x, tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)



# %% [markdown]
# ### Model Creation

# %%
def create_model(num_classes):
    """Create the CNN model."""
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# %% [markdown]
# ### Training Model

# %%
# Hyperparameters
SAMPLES_PER_CLASS = 10000
BATCH_SIZE = 100 * len(CATEGORIES)
EPOCHS = 1

# %%
def train_model(model, train_generator, val_generator, epochs):
    """Train the model using generators."""

    checkpoint_callback = ModelCheckpoint(
        filepath='model_checkpoint.keras',  # Path where the model is saved
        save_best_only=True,  # Save only the best model
        monitor='val_loss',  # Metric to monitor
        mode='min',  # Mode: 'min' for loss, 'max' for accuracy
        save_weights_only=False,  # Save full model (weights + architecture)
        verbose=1  # Verbosity mode
    )


    # optimizer = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam())
    
    model.compile(optimizer="adam",
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    
    history = model.fit(train_generator, 
                        validation_data=val_generator,
                        epochs=epochs,
                        callbacks=[checkpoint_callback] )
    return history

# %%
# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Configure TensorFlow to use memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Verify TensorFlow is using GPU
print("TensorFlow is using GPU:", tf.test.is_built_with_cuda())

# Enable TF32
# tf.config.experimental.enable_tensor_float_32_execution(True)

# Verify TF32 status
# print("TF32 enabled:", tf.config.experimental.tensor_float_32_execution_enabled())



# %%
 # Create data generators
train_generator = QuickDrawDataGenerator(DATA_DIR, CATEGORIES, SAMPLES_PER_CLASS, BATCH_SIZE, subset='train')
val_generator = QuickDrawDataGenerator(DATA_DIR, CATEGORIES, SAMPLES_PER_CLASS, BATCH_SIZE, subset='val')

# Create and train model
model = create_model(len(CATEGORIES))
history = train_model(model, train_generator, val_generator, EPOCHS)


# %% [markdown]
# ### Plotting & Accuracy

# %%
def plot_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# %%
# Plot training history
plot_history(history)

# %%
# Evaluate model
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation accuracy: {val_accuracy:.4f}")

# %% [markdown]
# ### Save Model

# %%
def save_model_keras_and_tflite(model, model_name):
    """
    Save the model in both Keras and TFLite formats.
    
    Args:
    model (tf.keras.Model): The trained model to save
    model_name (str): Base name for the saved model files
    """
    # Save in Keras format
    keras_path = f"{model_name}.h5"
    model.save(keras_path)
    print(f"Model saved in Keras format at: {keras_path}")
    

    # Convert to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.target_spec.supported_types = [tf.float16]

    # Enable TF Select ops
    #converter.target_spec.supported_ops = [
    #    tf.lite.OpsSet.TFLITE_BUILTINS,
    #    tf.lite.OpsSet.SELECT_TF_OPS
    #]

    tflite_model = converter.convert()

    # Save the TFLite model
    tflite_path = f"{model_name}.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved in TFLite format at: {tflite_path}")

    # Optionally: Convert to quantized TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    quantized_tflite_model = converter.convert()

    # Save the quantized TFLite model
    quantized_tflite_path = f"{model_name}_quantized.tflite"
    with open(quantized_tflite_path, 'wb') as f:
        f.write(quantized_tflite_model)
    print(f"Quantized model saved in TFLite format at: {quantized_tflite_path}")


# %%
save_model_keras_and_tflite(model, "model_cnn_sketch")

# %% [markdown]
# loaded_model = tf.keras.models.load_model("model_cnn_sketch.h5")
# 
# 
# 
# # Convert to TFLite format
# converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float32]
# 
# # Enable TF Select ops
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,
#     tf.lite.OpsSet.SELECT_TF_OPS
# ]
# 
# tflite_model = converter.convert()
# 
# # Save the TFLite model
# tflite_path = "model_cnn_sketch.tflite"
# with open(tflite_path, 'wb') as f:
#     f.write(tflite_model)
# print(f"Model saved in TFLite format at: {tflite_path}")

# %%



