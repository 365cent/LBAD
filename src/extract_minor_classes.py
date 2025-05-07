import tensorflow as tf
import os

def parse_tfrecord_fn(example):
    """Parse a TFRecord example into a dictionary of features."""
    feature_description = {
        'l': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    return example

def extract_minor_classes(tfrecord_path, label_map=None):
    import json
    if label_map is None:
        label_map = {0: "attacker_http", 1: "normal", 2: "service_scan"}
    idx_to_class = {k: v for k, v in label_map.items()}
    class_to_idx = {v: k for k, v in label_map.items()}
    extracted_logs = {"normal": [], "service_scan": []}
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")
    dataset = dataset.map(parse_tfrecord_fn)
    for record in dataset:
        log_line = record['l'].numpy().decode('utf-8')
        label_json = record['y'].numpy().decode('utf-8')
        try:
            labels = json.loads(label_json)
        except Exception:
            labels = []
        # If no label, treat as "normal"
        if not labels:
            if "normal" in extracted_logs:
                extracted_logs["normal"].append(log_line)
        else:
            for label in labels:
                if label in extracted_logs:
                    extracted_logs[label].append(log_line)
    return extracted_logs

def main():
    # Set the input directory to the processed directory
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processed')
    # Output directory (optional, remove if not needed)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    
    print(f"Reading TFRecord files from: {input_dir}")
    
    # Handle directory input
    tfrecord_paths = []
    for file in os.listdir(input_dir):
        if file.endswith('.tfrecord'):
            tfrecord_paths.append(os.path.join(input_dir, file))
    
    if not tfrecord_paths:
        print(f"No TFRecord files found in {input_dir}")
        return
        
    all_extracted_logs = {"normal": [], "service_scan": []}
    
    # Process each TFRecord file
    for path in tfrecord_paths:
        print(f"Processing: {path}")
        try:
            extracted_logs = extract_minor_classes(path)
            # Merge results
            for cls in all_extracted_logs.keys():
                all_extracted_logs[cls].extend(extracted_logs[cls])
        except tf.errors.DataLossError as e:
            print(f"WARNING: Skipping corrupted TFRecord file: {path}\nError: {e}")
    
    # Print statistics and sample logs
    for cls_name, logs in all_extracted_logs.items():
        print(f"\n{'=' * 50}")
        print(f"Class: {cls_name} - {len(logs)} records found")
        print(f"{'=' * 50}")
        
        # Print sample logs (first 10)
        sample_size = min(10, len(logs))
        for i in range(sample_size):
            print(f"{i+1}. {logs[i]}")
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{cls_name}_logs.txt")
        with open(output_file, 'w') as f:
            for log in logs:
                f.write(f"{log}\n")
        print(f"Saved {len(logs)} {cls_name} logs to: {output_file}")

if __name__ == "__main__":
    main()
