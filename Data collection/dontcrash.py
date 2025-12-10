import pandas as pd

# Path to the CSV file
file_path = r"DS-Project\STUFFALOTSOFSTUFF\nasdaq_exteral_data.csv"

# Read in chunks and only get the first 20 lines
chunksize = 1000  # Read 1000 rows at a time
first_20_lines = None
lines_read = 0

print(f"Reading first 20 lines from {file_path}...")
print("-" * 80)

try:
    # Read CSV in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        if first_20_lines is None:
            # First chunk - take first 20 lines
            if len(chunk) >= 20:
                first_20_lines = chunk.head(20)
                lines_read = 20
                break
            else:
                first_20_lines = chunk.copy()
                lines_read = len(chunk)
        else:
            # Append to existing data until we have 20 lines
            remaining = 20 - lines_read
            if remaining > 0:
                first_20_lines = pd.concat([first_20_lines, chunk.head(remaining)], ignore_index=True)
                lines_read += len(chunk.head(remaining))
                if lines_read >= 20:
                    break
            else:
                break
    
    if first_20_lines is not None:
        print(f"Successfully read {len(first_20_lines)} lines")
        print(f"\nShape: {first_20_lines.shape}")
        print(f"\nColumns: {list(first_20_lines.columns)}")
        print("\n" + "=" * 80)
        print("First 20 lines:")
        print("=" * 80)
        print(first_20_lines.to_string())
        print("\n" + "=" * 80)
        print("\nData types:")
        print(first_20_lines.dtypes)
    else:
        print("No data read from file")
        
except Exception as e:
    print(f"Error reading file: {e}")
    import traceback
    traceback.print_exc()