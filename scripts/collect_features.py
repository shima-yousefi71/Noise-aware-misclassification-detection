import os
import pickle
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.datasets import cifar100
from keras.applications.vgg19 import preprocess_input
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and memory growth is set.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available. The code will run on CPU.")
np.random.seed(5)
tf.random.set_seed(5)

# Load CIFAR-100 data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Shuffle the dataset
indices = np.random.permutation(x_train.shape[0])
x_train = x_train[indices]
y_train = y_train[indices]

# Collect only a subset of samples (e.g., 5000)
num_samples = 10000
x_train_subset = x_train[:num_samples]
y_train_subset = y_train[:num_samples]

# Resize images to 224x224 as expected by VGG19
x_train_resized = np.array([tf.image.resize(image, (224, 224)).numpy() for image in x_train_subset])
x_test_resized = np.array([tf.image.resize(image, (224, 224)).numpy() for image in x_test])

# Preprocess images
x_train_resized = preprocess_input(x_train_resized)
x_test_resized = preprocess_input(x_test_resized)

# Load the retrained model
retrained_model = load_model('models/final_model_cifar100.h5')
print(retrained_model.summary())

# Define the intermediate layer model
layer_indices = [20]  # Layers to extract outputs from
intermediate_layer_model = Model(
    inputs=retrained_model.input,
    outputs=[retrained_model.layers[idx].output for idx in layer_indices]
)

# Directory to save the intermediate outputs
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Function to save intermediate outputs
def save_layer_outputs(output, layer_index, class_label, sample_index):
    layer_output_dir = os.path.join(output_dir, f"layer_{layer_index}")
    os.makedirs(layer_output_dir, exist_ok=True)
    output_file = os.path.join(
        layer_output_dir, f"sample_{sample_index}_class_{class_label}_layer_{layer_index}.pkl"
    )
    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

# Process samples and save intermediate outputs
for i in range(num_samples):
    # Get intermediate outputs for the current sample
    intermediate_outputs = intermediate_layer_model.predict(x_train_resized[i:i+1])

    # Save outputs for each specified layer
    for layer_idx, layer_output in zip(layer_indices, intermediate_outputs):
        save_layer_outputs(
            layer_output, 
            layer_index=layer_idx, 
            class_label=y_train_subset[i][0],  # True class label
            sample_index=i  # Sample index
        )
    if i % 100 == 0:
        print(f"Processed {i}/{num_samples} samples")

print("Intermediate outputs collected and saved successfully.")

