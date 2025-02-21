import pandas as pd
import numpy as np
import ast  # To safely evaluate string representations of lists

def feature_selection_correlation(data, threshold=0.8):
    """
    Perform feature selection based on correlation.

    Args:
        data (pd.DataFrame): Input data with numerical features.
        threshold (float): Correlation threshold for feature removal.

    Returns:
        pd.DataFrame: Reduced data with selected features.
        list: Indices of selected features.
    """
    # Compute the correlation matrix
    correlation_matrix = data.corr()
    
    selected_features = list(range(data.shape[1]))
    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            if abs(correlation_matrix.iloc[i, j]) > threshold and j in selected_features:
                selected_features.remove(j)
    
    reduced_data = data.iloc[:, selected_features]
    return reduced_data, selected_features

def preprocess_data(input_file):
    """
    Preprocess the data to ensure compatibility with correlation calculations.

    Args:
        input_file (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Processed data ready for feature selection.
        pd.Series: Labels (if present).
    """
    # Load raw data
    data = pd.read_csv(input_file)

    # Parse the 'Sequence' column if it exists
    if 'Sequence' in data.columns:
        data['Sequence'] = data['Sequence'].apply(ast.literal_eval)  # Convert string to list of lists
        # Flatten the 'Sequence' column into individual feature columns
        sequences = pd.DataFrame(data['Sequence'].tolist(), columns=[f"seq_{i}" for i in range(len(data['Sequence'].iloc[0]))])
        data = pd.concat([sequences, data.drop(columns=['Sequence'])], axis=1)
    
    # Separate labels if present
    labels = data["Label"] if "Label" in data.columns else None
    features = data.drop(columns=["Label"]) if "Label" in data.columns else data

    return features, labels

def main():
    # File paths
    input_file = "converted_monday.csv"
    output_file = "filtered_data1.csv"

    # Preprocess the data
    features, labels = preprocess_data(input_file)

    # Apply feature selection
    filtered_data, selected_features = feature_selection_correlation(features, threshold=0.8)

    # Combine filtered features and labels (if labels exist)
    if labels is not None:
        filtered_data["Label"] = labels.reset_index(drop=True)
    
    # Save the filtered data
    filtered_data.to_csv(output_file, index=False)

    print(f"Feature selection complete. Filtered data saved to {output_file}.")
    print(f"Selected feature indices: {selected_features}")

if __name__ == "__main__":
    main()
