import pandas as pd

def split_csv(file_path, output_file1, output_file2):
    """
    Split a CSV file into two equal halves and save as separate files.

    Args:
        file_path (str): Path to the input CSV file.
        output_file1 (str): Path for the first half of the CSV.
        output_file2 (str): Path for the second half of the CSV.

    Returns:
        None
    """
    # Read the CSV file
    data = pd.read_csv(file_path)
    print(f"Loaded data with {len(data)} rows.")

    # Find the split index
    split_index = len(data) // 2

    # Split the data
    first_half = data.iloc[:split_index]
    second_half = data.iloc[split_index:]

    # Save the halves as separate CSV files
    first_half.to_csv(output_file1, index=False)
    second_half.to_csv(output_file2, index=False)

    print(f"First half saved to {output_file1} with {len(first_half)} rows.")
    print(f"Second half saved to {output_file2} with {len(second_half)} rows.")

# Example usage
split_csv("sequences_monday_hirst_half.csv", "sequences_monday_hirst_half_imp2.csv", "sequences_monday_second_half_imp2.csv")
