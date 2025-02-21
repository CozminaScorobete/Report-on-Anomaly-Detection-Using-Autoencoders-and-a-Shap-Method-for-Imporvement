import pandas as pd

def convert_labels(input_file, output_file):
    """
    Convert 'BENIGN' to 0 and 'DDoS' to 1 in the dataset labels, and save the updated CSV.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the updated CSV file.
    """
    # Load the dataset
    data = pd.read_csv(input_file)

    # Normalize column names
    data.columns = data.columns.str.strip()

    # Ensure 'Label' column exists
    if 'Label' not in data.columns:
        raise ValueError("'Label' column not found in the dataset.")

    # Normalize and map labels
    data['Label'] = data['Label'].str.strip().str.upper()  # Normalize case and strip spaces
    label_mapping = {'BENIGN': 0, 'DDOS': 1}
    data['Label'] = data['Label'].map(label_mapping)

    # Drop rows with unmapped labels (if any)
    print("Labels before dropping unmapped:", data['Label'].unique())
    data = data.dropna(subset=['Label'])
    print("Labels after dropping unmapped:", data['Label'].unique())

    # Save the updated dataset
    data.to_csv(output_file, index=False)
    print(f"Converted labels and saved the file to {output_file}.")

def create_sequences(input_file, output_file, sequence_length=10):
    """
    Create sequences of a fixed length from the dataset and save to a new CSV file.

    Args:
        input_file (str): Path to the input CSV file with converted labels.
        output_file (str): Path to save the sequences.
        sequence_length (int): Length of each sequence.
    """
    # Load the dataset
    data = pd.read_csv(input_file)

    # Separate features and labels
    features = data.drop(columns=['Label']).values
    labels = data['Label'].values

    # Create sequences
    sequences = []
    sequence_labels = []
    for i in range(len(features) - sequence_length + 1):
        sequences.append(features[i:i + sequence_length])
        sequence_labels.append(labels[i + sequence_length - 1])

    # Save sequences as a new dataset
    sequence_data = {
        'Sequence': [seq.tolist() for seq in sequences],  # Convert to list for CSV compatibility
        'Label': sequence_labels
    }
    sequence_df = pd.DataFrame(sequence_data)
    sequence_df.to_csv(output_file, index=False)
    print(f"Created sequences and saved to {output_file}.")

def main():
    # File paths
    monday_file = "_mondey_m2.csv"
    friday_file = "DDoS-Friday-no-metadata.csv"
    converted_monday_file = "converted_monday.csv"
    converted_friday_file = "converted_friday.csv"
    sequences_monday_file = "sequences_monday_m2.csv"
    sequences_friday_file = "sequences_friday_1.csv"

    # Convert labels for Monday and Friday data
    print("Converting labels in Monday data...")
    #convert_labels(monday_file, converted_monday_file)

    print("Converting labels in Friday data...")
    #convert_labels(friday_file, converted_friday_file)

    # Create sequences for Monday and Friday data
    print("Creating sequences for Monday data...")
    create_sequences(converted_monday_file, sequences_monday_file)

    print("Creating sequences for Friday data...")
    #create_sequences(converted_friday_file, sequences_friday_file)

if __name__ == "__main__":
    main()
