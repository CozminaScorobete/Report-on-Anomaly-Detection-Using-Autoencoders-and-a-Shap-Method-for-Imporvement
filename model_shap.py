import numpy as np
import pandas as pd
import shap
import ast
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

def preprocess_data(sequence_file):
    """
    Preprocess sequence data with proper shape handling
    """
    data = pd.read_csv(sequence_file)
    data['Sequence'] = data['Sequence'].apply(ast.literal_eval)
    sequences = np.array(data['Sequence'].tolist())
    labels = data['Label'].values
    
    # Flatten sequences while preserving samples
    sequences = sequences.reshape(sequences.shape[0], -1)
    input_shape = (sequences.shape[1],)
    
    # Scale data
    scaler = StandardScaler()
    sequences = scaler.fit_transform(sequences)
    
    return sequences, labels, input_shape, scaler

def build_autoencoder(input_shape, encoding_dim=10):
    """
    Build autoencoder with specified input shape and encoding dimension
    """
    input_dim = input_shape[0]
    
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Encoder layers
    encoded = Dense(input_dim // 2, activation='relu')(input_layer)
    encoded = Dense(input_dim // 4, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder layers
    decoded = Dense(input_dim // 4, activation='relu')(encoded)
    decoded = Dense(input_dim // 2, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    # Create and compile model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return autoencoder

def apply_shap(autoencoder, data, background_sample_size=10, n_top_features=20):
    """
    Apply SHAP analysis for feature selection
    """
    print("Starting SHAP analysis...")
    
    def predict_fn(X):
        """Prediction function for SHAP that returns reconstruction error"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        reconstructed = autoencoder.predict(X)
        # Calculate MSE across features only
        reconstruction_error = np.mean(np.square(X - reconstructed), axis=1)
        return reconstruction_error
    
    # Select background data
    background_data = data[:background_sample_size]
    
    print(f"Background data shape: {background_data.shape}")
    print("Initializing SHAP explainer...")
    
    # Initialize explainer
    explainer = shap.KernelExplainer(
        predict_fn,
        background_data,
        link="identity"
    )
    
    # Calculate SHAP values
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(
        background_data,
        nsamples=50,
        l1_reg="num_features(10)"
    )
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Calculate feature importance
    feature_importance = np.abs(shap_values).mean(0)
    top_features = np.argsort(feature_importance)[-n_top_features:]
    
    return top_features, feature_importance

def train_shap_model(train_file, model_file, epochs=100, batch_size=32):
    """
    Train autoencoder with SHAP-based feature selection
    """
    # Load and preprocess data
    sequences, labels, input_shape, scaler = preprocess_data(train_file)
    print(f"Initial data shape: {sequences.shape}")
    
    # Train initial model
    initial_autoencoder = build_autoencoder(input_shape)
    history = initial_autoencoder.fit(
        sequences, 
        sequences,
        epochs=100,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Apply SHAP analysis
    print("\nApplying SHAP analysis...")
    top_features, feature_importance = apply_shap(
        initial_autoencoder, 
        sequences,
        background_sample_size=10
    )
    
    # Select top features
    selected_features = sequences[:, top_features]
    
    # Build and train final model
    final_shape = (len(top_features),)
    final_autoencoder = build_autoencoder(final_shape)
    
    print("\nTraining final model...")
    final_autoencoder.fit(
        selected_features,
        selected_features,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    final_autoencoder.save(model_file)
    return final_autoencoder, top_features, scaler

def evaluate_model(model_file, test_file, top_features, threshold=0.5):
    """
    Evaluate model performance
    """
    autoencoder = load_model(model_file)
    sequences, labels, _, _ = preprocess_data(test_file)
    
    # Select features
    selected_features = sequences[:, top_features]
    
    # Get predictions
    reconstructed = autoencoder.predict(selected_features)
    mse = np.mean(np.square(selected_features - reconstructed), axis=1)
    predictions = (mse > threshold).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(predictions == labels)
    precision = np.sum((predictions == 1) & (labels == 1)) / (np.sum(predictions == 1) + 1e-10)
    recall = np.sum((predictions == 1) & (labels == 1)) / (np.sum(labels == 1) + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return predictions, mse

def main():
    train_file = "data.csv"
    test_file = "data1.csv"
    model_file = "shap_autoencoder.h5"
    
    # Train model
    print("Training model...")
    model, top_features, scaler = train_shap_model(train_file, model_file)
    
    # Evaluate model
    print("\nEvaluating model...")
    predictions, reconstruction_errors = evaluate_model(
        model_file, 
        test_file, 
        top_features
    )

if __name__ == "__main__":
    main()