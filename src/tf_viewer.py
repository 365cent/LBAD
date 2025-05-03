import tensorflow as tf
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import TclError

def parse_tfrecord(tfrecord_path, max_records=50):
    """Parse a TFRecord file and return a list of (log_line, labels) tuples."""
    dataset = tf.data.TFRecordDataset([str(tfrecord_path)], compression_type='GZIP')
    feature_description = {
        'l': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
    }
    
    records = []
    for raw_record in dataset.take(max_records):  # Limit to max_records to avoid huge images
        example = tf.io.parse_single_example(raw_record, feature_description)
        log_line = example['l'].numpy().decode('utf-8', errors='replace')
        labels_json = example['y'].numpy().decode('utf-8', errors='replace')
        try:
            labels = json.loads(labels_json)
            labels_str = json.dumps(labels, indent=2)
        except json.JSONDecodeError:
            labels_str = "Invalid JSON"
        records.append((log_line, labels_str))
    return records

def create_image(records, output_path, title):
    """Create an image with a two-column display of log lines and labels."""
    # Image configuration
    width = 1200  # Total width of the image
    column_width = width // 2 - 20  # Width per column (with padding)
    padding = 20
    line_height = 20
    font_size = 14
    max_chars_per_line = 50  # Approx characters before wrapping

    # Estimate image height
    total_lines = sum(
        max(len(log_line) // max_chars_per_line + 1, len(labels.split('\n')))
        for log_line, labels in records
    )
    height = total_lines * line_height + 100  # Extra for title and padding

    # Create image
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Load a monospaced font
    try:
        font = ImageFont.truetype("Courier New", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Draw title
    draw.text((padding, padding), title, fill='black', font=font)
    y_offset = padding + 40

    # Draw headers
    draw.text((padding, y_offset), "Log Line", fill='blue', font=font)
    draw.text((width // 2 + padding, y_offset), "Labels", fill='blue', font=font)
    y_offset += line_height * 2

    # Draw each record
    for log_line, labels in records:
        # Wrap log line
        log_lines = []
        current_line = ""
        for char in log_line:
            current_line += char
            if len(current_line) >= max_chars_per_line:
                log_lines.append(current_line)
                current_line = ""
        if current_line:
            log_lines.append(current_line)

        # Split labels by lines
        label_lines = labels.split('\n')
        max_lines = max(len(log_lines), len(label_lines))

        # Draw log lines
        for i in range(max_lines):
            log_text = log_lines[i] if i < len(log_lines) else ""
            label_text = label_lines[i] if i < len(label_lines) else ""
            
            # Draw log line
            draw.text((padding, y_offset), log_text, fill='black', font=font)
            # Draw label
            draw.text((width // 2 + padding, y_offset), label_text, fill='black', font=font)
            y_offset += line_height

        y_offset += line_height  # Extra spacing between records

    # Save the image
    image.save(output_path, 'PNG')
    return output_path

def main():
    # Set the directory containing the TFRecord files
    output_dir = Path(__file__).resolve().parent.parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir = output_dir / "viewer_images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # Find all TFRecord files
    tfrecord_files = list(output_dir.glob("*.tfrecord"))

    if not tfrecord_files:
        print("No TFRecord files found.")
        return

    # Process each TFRecord file
    for tfrecord_file in tfrecord_files:
        print(f"Processing {tfrecord_file}...")
        records = parse_tfrecord(tfrecord_file)
        if not records:
            print(f"No records found in {tfrecord_file}")
            continue

        # Generate image
        image_path = image_output_dir / f"{tfrecord_file.stem}_viewer.png"
        title = f"TFRecord Viewer: {tfrecord_file.name}"
        created_image = create_image(records, image_path, title)
        print(f"Generated image: {created_image}")

if __name__ == "__main__":
    main()