import pandas as pd
import os
import re
import datetime

def parse_transcript(file_path):
    """
    Parses a transcript file and extracts relevant sections into a structured format.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    current_speaker = None
    current_time = None
    current_text = []

    for line in lines:
        line = line.strip()

        # Match speaker + timestamp pattern
        match = re.match(r'(\[redacted\]|\w+ \w+)\s+(\d{2}:\d{2})', line)
        if match:
            # Save previous entry if exists
            if current_speaker and current_text:
                data.append([current_speaker, current_time, " ".join(current_text)])
            
            # Extract new speaker and timestamp
            current_speaker = match.group(1)
            current_time = match.group(2)
            current_text = [line[len(match.group(0)):].strip()]  # Capture text after timestamp

        elif current_speaker:
            # Continuation of previous speaker's text
            current_text.append(line)

    # Save last entry
    if current_speaker and current_text:
        data.append([current_speaker, current_time, " ".join(current_text)])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Speaker", "Timestamp", "Text"])

    return df

def save_to_csv(df, original_file_path):
    """
    Saves the parsed DataFrame to a uniquely named CSV file.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"parsed_{timestamp}.csv"
    output_path = os.path.join(os.path.dirname(original_file_path), output_filename)
    
    df.to_csv(output_path, index=False)
    print(f"Parsed transcript saved to: {output_path}")

# Example usage
file_path = 'data/CoP_Recording_3.11.21.txt'  # Update this for new files
df = parse_transcript(file_path)
save_to_csv(df, file_path)
