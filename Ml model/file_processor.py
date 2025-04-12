import pandas as pd

def clean_and_structure_data(file_path, output_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        df = df.astype(str)  # Convert all columns to string
        df = df.dropna(how='all')  # Drop rows with all NaN
        df.to_csv(output_path, index=False)
        return df.to_dict('records')
    except pd.errors.ParserError:
        return None
    except Exception:
        return None

def extract_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return "Error reading file"