from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import ast  # Import for safer string evaluation
import matplotlib.pyplot as plt

def build_autoencoder(input_dim):
    print(f"Building autoencoder with input dimension: {input_dim}")
    inputs = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(70, activation='relu')(inputs)
    encoded = Dense(30, activation='relu')(encoded)
    encoded = Dense(10, activation='relu')(encoded)  # Bottleneck layer

    # Decoder
    decoded = Dense(30, activation='relu')(encoded)
    decoded = Dense(70, activation='relu')(decoded)
    outputs = Dense(input_dim)(decoded)

    # Autoencoder
    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    print("Autoencoder built and compiled.")
    return autoencoder


def preprocess_data(sequence_file):
    print(f"Preprocessing data from file: {sequence_file}")
    data = pd.read_csv(sequence_file)
    print(f"Data loaded. First 5 rows:\n{data.head()}")

    # Parse the 'Sequence' column from string to list of lists
    data['Sequence'] = data['Sequence'].apply(ast.literal_eval)  # Safer parsing of string to Python list
    sequences = np.array(data['Sequence'].tolist())  # Convert to NumPy array
    labels = data['Label'].values

    print(f"Parsed sequences shape: {sequences.shape}")
    print(f"Sample sequence: {sequences[0]}")

    # Normalize sequences
    scaler = MinMaxScaler()
    n_samples, timesteps, n_features = sequences.shape
    sequences = sequences.reshape(sequences.shape[0], -1)  # Flatten the sequences
    print(f"Flattened sequences shape: {sequences.shape}")
    sequences = scaler.fit_transform(sequences)  # Normalize flattened data
    print(f"Normalized sequences shape: {sequences.shape}")

    return sequences, labels


def train_autoencoder(sequence_file, model_file, epochs=100, batch_size=8192):
    print(f"Training autoencoder on data from: {sequence_file}")
    # Preprocess the data
    sequences, _ = preprocess_data(sequence_file)

    print(f"Training data shape: {sequences.shape}")

    # Build and train the autoencoder
    autoencoder = build_autoencoder(input_dim=sequences.shape[1])
    print("Starting training...")
    autoencoder.fit(sequences, sequences, epochs=epochs, batch_size=batch_size, validation_split=0.33)
    autoencoder.save(model_file)
    print(f"Model saved to {model_file}")
    return autoencoder


def test_autoencoder(model_file, sequence_file, threshold):
    print(f"Testing autoencoder using model from: {model_file}")
    print(f"Test data from: {sequence_file}")
    print(f"Using reconstruction error threshold: {threshold}")
    
    # Load the trained model
    autoencoder = load_model(model_file)
    print("Model loaded successfully.")

    # Preprocess the test data
    sequences, labels = preprocess_data(sequence_file)

    print(f"Test data shape: {sequences.shape}")
    print(f"Labels distribution: {np.bincount(labels)}")

    # Predict reconstruction errors
    reconstructed = autoencoder.predict(sequences)
    print("Reconstruction completed.")
    reconstruction_errors = np.mean(np.square(sequences - reconstructed), axis=1)
    print(f"Reconstruction errors (first 10): {reconstruction_errors[:10]}")

    # Classify based on the threshold
    predictions = (reconstruction_errors > threshold).astype(int)

    print(f"Predictions (first 10): {predictions[:10]}")
    print(f"Labels (first 10): {labels[:10]}")

    # Compute evaluation metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    auc = roc_auc_score(labels, reconstruction_errors)

    print("Testing Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")


def plot_reconstruction_errors(errors, labels):
    normal_errors = errors[labels == 0]
    anomaly_errors = errors[labels == 1]

    plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red')
    plt.axvline(x=threshold, color='green', linestyle='--', label='Threshold')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Example usage


def main():
    # File paths
    train_file = "data.csv"
    test_file = "data1.csv"
    model_file = "autoencoder_model_m1.h5"

    # Train the autoencoder
    print("Training the autoencoder...")
    train_autoencoder(train_file, model_file)

    # Test the autoencoder
    print("Testing the autoencoder...")
    test_autoencoder(model_file, test_file, threshold=0.17)
   

if __name__ == "__main__":
    main()
