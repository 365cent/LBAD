import tensorflow as tf
import pandas as pd
import json
from pathlib import Path

def parse_example(example):
    """Parse a TensorFlow Example protocol buffer."""
    feature_description = {
        'l': tf.io.FixedLenFeature([], tf.string),  # log
        'y': tf.io.FixedLenFeature([], tf.string),  # label
    }
    return tf.io.parse_single_example(example, feature_description)

def summarize_tfrecords(directory="processed", sample_size=3):
    """Display summary of all TFRecord files in the directory."""
    tfrecord_files = list(Path(directory).glob("**/*.tfrecord"))
    
    if not tfrecord_files:
        print(f"No TFRecord files found in {directory}")
        return
    
    print(f"Found {len(tfrecord_files)} TFRecord files")
    print("-" * 60)
    
    for file_path in tfrecord_files:
        try:
            # Load dataset
            dataset = tf.data.TFRecordDataset(str(file_path), compression_type="GZIP")
            
            # Count records
            count = sum(1 for _ in dataset)
            
            # Display file summary
            print(f"File: {file_path.name}")
            print(f"Records: {count}")
            
            # Show sample records as a table
            print("\n| # | Log | Label |")
            print("|---|" + "-" * 53 + "|" + "-" * 30 + "|")
            
            for i, raw_record in enumerate(dataset.take(sample_size)):
                parsed = parse_example(raw_record)
                log = parsed['l'].numpy().decode('utf-8')
                label = parsed['y'].numpy().decode('utf-8')
                
                # Truncate log but show full label
                log_display = (log[:50] + "...") if len(log) > 50 else log
                
                # Format as table row
                print(f"| {i+1} | {log_display} | {label} |")
            
            print("-" * 60)
        
        except Exception as e:
            print(f"Error with {file_path.name}: {e}")
            print("-" * 60)

if __name__ == "__main__":
    summarize_tfrecords()