
from google.colab import drive
drive.mount('/content/drive')

!pip install pyswarm
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pyswarm import pso

# Define file paths for data
data_dir = '/content/drive/MyDrive/Dataset'

import os
from sklearn.model_selection import train_test_split
from shutil import copyfile

def auto_split_dataset(input_folder, output_folder):
    # Create output folders if not exist
    for folder in ['train', 'test', 'val']:
        os.makedirs(os.path.join(output_folder, folder, 'fire'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, folder, 'nofire'), exist_ok=True)

    # Get the list of image filenames
    fire_images = os.listdir(os.path.join(input_folder, 'fire'))
    nofire_images = os.listdir(os.path.join(input_folder, 'nofire'))

    # Split the dataset into training, testing, and validation sets
    fire_train, fire_temp = train_test_split(fire_images, test_size=0.3, random_state=42)
    fire_test, fire_val = train_test_split(fire_temp, test_size=0.5, random_state=42)

    nofire_train, nofire_temp = train_test_split(nofire_images, test_size=0.3, random_state=42)
    nofire_test, nofire_val = train_test_split(nofire_temp, test_size=0.5, random_state=42)

    # Copy images to respective folders
    for image in fire_train:
        copyfile(os.path.join(input_folder, 'fire', image), os.path.join(output_folder, 'train', 'fire', image))

    for image in fire_test:
        copyfile(os.path.join(input_folder, 'fire', image), os.path.join(output_folder, 'test', 'fire', image))

    for image in fire_val:
        copyfile(os.path.join(input_folder, 'fire', image), os.path.join(output_folder, 'val', 'fire', image))

    for image in nofire_train:
        copyfile(os.path.join(input_folder, 'nofire', image), os.path.join(output_folder, 'train', 'nofire', image))

    for image in nofire_test:
        copyfile(os.path.join(input_folder, 'nofire', image), os.path.join(output_folder, 'test', 'nofire', image))

    for image in nofire_val:
        copyfile(os.path.join(input_folder, 'nofire', image), os.path.join(output_folder, 'val', 'nofire', image))

# Example usage:
input_folder = '/content/drive/MyDrive/Dataset'
output_folder = '/content/drive/MyDrive/Output'
auto_split_dataset(input_folder, output_folder)

# Get the list of classes dynamically
import os
classes = ['fire', 'nofire']

# Create ImageDataGenerator for data augmentation and normalization
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.3)

# Load and split the data into training, validation, and testing sets
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    classes=classes,
    batch_size=32,
    shuffle=True,
    subset='training'
)
val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    classes=classes,
    batch_size=32,
    shuffle=True,
    subset='validation'
)

# Create InceptionV3 model with pre-trained weights
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom head to the base model
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define objective function for Hybrid PSO
def objective_function(x):
    # Set the hyperparameters based on the particle values
    learning_rate = x[0]
    num_epochs = math.ceil(x[1])

    # Train the model with the updated hyperparameters
    model.fit(train_generator, validation_data=val_generator, epochs=num_epochs, verbose=0)

    # Obtain the accuracy on the validation set
    _, val_acc = model.evaluate(val_generator)

    # Return the negative accuracy to maximize it in PSO
    return -val_acc

# Define Hybrid PSO parameters
def hybrid_pso(objective_function, lower_bounds, upper_bounds, max_iterations=50, swarm_size=10):
    inertia_weight = 0.5
    c1 = 1.5
    c2 = 1.5
    bounds = list(zip(lower_bounds, upper_bounds))

    # Initialize particles and velocities
    particles = np.random.uniform(lower_bounds, upper_bounds, (swarm_size, len(lower_bounds)))
    velocities = np.random.uniform(-1, 1, (swarm_size, len(lower_bounds)))

    # Initialize personal best positions and fitness values
    personal_best_positions = particles.copy()
    personal_best_fitness = np.array([objective_function(p) for p in particles])

    # Initialize global best position and fitness value
    global_best_index = np.argmin(personal_best_fitness)
    global_best_position = particles[global_best_index]
    global_best_fitness = personal_best_fitness[global_best_index]

    # Initialize memory matrix
    memory_matrix = np.zeros_like(particles)

    for iteration in range(max_iterations):
        for i in range(swarm_size):
            # Update particle velocity using the Hybrid PSO rule
            r1, r2 = np.random.rand(), np.random.rand()

            cognitive_component = c1 * r1 * (personal_best_positions[i] - particles[i])
            social_component = c2 * r2 * (global_best_position - particles[i])
            memory_component = inertia_weight * memory_matrix[i]

            velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component + memory_component

            # Update particle position
            particles[i] += velocities[i]

            # Ensure particle stays within bounds
            particles[i] = np.clip(particles[i], lower_bounds, upper_bounds)

            # Evaluate fitness of the new position
            fitness = objective_function(particles[i])

            # Update personal best if needed
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best_positions[i] = particles[i]

                # Update global best if needed
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particles[i]

        # Update memory matrix
        memory_matrix = personal_best_positions.copy()

        # Update inertia weight (self-adaptation)
        inertia_weight = max(0.4, inertia_weight - 0.002)

    return global_best_position, global_best_fitness

# Set the bounds for the hyperparameters
lower_bounds = [0.001, 2]  # Lower bounds for learning_rate and num_epochs
upper_bounds = [0.003, 3]  # Upper bounds for learning_rate and num_epochs
swarm_size = 2  # Adjust the number of particles

# Perform Hybrid PSO optimization
best_solution, best_fitness = hybrid_pso(objective_function, lower_bounds, upper_bounds, max_iterations=2, swarm_size=swarm_size)


# Print the best solution and its fitness
print("Best Solution (Learning Rate, Num Epochs):", best_solution)
print("Best Fitness (Validation Accuracy):", -best_fitness)

# Train the model with the best hyperparameters
best_num_epochs = math.ceil(best_solution[1])
history = model.fit(train_generator, validation_data=val_generator, epochs=best_num_epochs)

# Evaluate the model on the validation set
val_generator.reset()
val_pred_labels = np.argmax(model.predict(val_generator), axis=1)
val_true_labels = val_generator.classes

# Calculate classification report and confusion matrix
report = classification_report(val_true_labels, val_pred_labels, target_names=classes)
confusion_mat = confusion_matrix(val_true_labels, val_pred_labels)

# Plot training and validation confusion matrices with heatmaps
plt.imshow(confusion_mat, cmap='Blues')
plt.xticks(np.arange(len(classes)), classes, rotation=45)
plt.yticks(np.arange(len(classes)), classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Validation Confusion Matrix')
plt.show()

# Print classification report and confusion matrix
print("Classification Report:")
print(report)
print("\nConfusion Matrix:")
print(confusion_mat)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# Calculate Precision, Recall, F1-Score
precision = precision_score(val_true_labels, val_pred_labels)
recall = recall_score(val_true_labels, val_pred_labels)
f1 = f1_score(val_true_labels, val_pred_labels)

# Calculate True Positives, False Positives, True Negatives, False Negatives
tp = confusion_mat[1, 1]
fp = confusion_mat[0, 1]
tn = confusion_mat[0, 0]
fn = confusion_mat[1, 0]

# Print the metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("True Positives:", tp)
print("False Positives:", fp)
print("True Negatives:", tn)
print("False Negatives:", fn)

# Plot the training and validation accuracy graph
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Plot the training and validation loss graph
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model on the training set
train_generator.reset()
train_pred_labels = np.argmax(model.predict(train_generator), axis=1)
train_true_labels = train_generator.classes

# Calculate confusion matrix for the training set
train_confusion_mat = confusion_matrix(train_true_labels, train_pred_labels)

# Plot training confusion matrix
plt.imshow(train_confusion_mat, cmap='Blues')
plt.xticks(np.arange(len(classes)), classes, rotation=45)
plt.yticks(np.arange(len(classes)), classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Training Confusion Matrix')
plt.show()

# Plot validation confusion matrix
plt.imshow(confusion_mat, cmap='Blues')
plt.xticks(np.arange(len(classes)), classes, rotation=45)
plt.yticks(np.arange(len(classes)), classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Validation Confusion Matrix')
plt.show()

# Calculate the number of images per class in the training set
train_class_counts = np.sum(train_generator.labels == np.arange(len(classes))[:, None], axis=1)

# Calculate the number of images per class in the validation set
val_class_counts = np.sum(val_generator.labels == np.arange(len(classes))[:, None], axis=1)

# Create a line plot for the number of images per class
plt.plot(classes, train_class_counts, label='Training Set', marker='o')
plt.plot(classes, val_class_counts, label='Validation Set', marker='o')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.legend()
plt.title('Number of Images per Class')
plt.show()

# Assuming you have 2 classes, get probabilities for the positive class (fire)
val_pred_probs_fire = val_pred_probs[:, 1]

# Calculate ROC curves and AUC for each class
fpr, tpr, thresholds = roc_curve(val_true_labels, val_pred_probs_fire)
roc_auc = roc_auc_score(val_true_labels, val_pred_probs_fire)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve for Fire Detection')
plt.show()

# Calculate error rates for each class
threshold = 0.5  # Adjust the threshold based on your requirements
val_pred_labels = (val_pred_probs_fire > threshold).astype(int)
error_rates = np.mean(val_pred_labels != val_true_labels, axis=0)

# Ensure error_rates is a 1D array
error_rates = np.squeeze(error_rates)

# Print error rates for each class
for i, class_name in enumerate(classes):
    print(f"Error Rate for {class_name}: {error_rates:.4f}")

# Calculate the number of images per class in the training set
train_class_counts = np.sum(train_generator.labels == np.arange(len(classes))[:, None], axis=1)

# Calculate the number of images per class in the validation set
val_class_counts = np.sum(val_generator.labels == np.arange(len(classes))[:, None], axis=1)

# Create a bar plot for the number of images per class
plt.bar(classes, train_class_counts, label='Training Set', alpha=0.7)
plt.bar(classes, val_class_counts, label='Validation Set', alpha=0.7)
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.legend()
plt.title('Number of Images per Class')
plt.show()