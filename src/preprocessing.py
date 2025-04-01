import json
import logging
import os
import mimetypes
from pathlib import Path
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LogPreprocessor")

# Initialize mimetypes database
mimetypes.init()

class LogPreprocessor:
    def __init__(self, logs_dir=None, labels_dir=None, output_dir=None):
        # Set base directory to the project root (parent of the directory containing this file)
        base_dir = Path(__file__).resolve().parent.parent
        
        self.logs_dir = base_dir / "logs" if logs_dir is None else Path(logs_dir)
        self.labels_dir = base_dir / "labels" if labels_dir is None else Path(labels_dir)
        self.output_dir = base_dir / "processed" if output_dir is None else Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_text_file(self, file_path):
        """Determine if a file is a text file efficiently."""
        # First check MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type.startswith('text/'):
            return True
            
        # Fall back to content inspection
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(1024)  # Sample first 1KB
                if b'\0' in sample:
                    return False
                try:
                    sample.decode('utf-8')
                    return True
                except UnicodeDecodeError:
                    return bool(sample.decode('latin-1', errors='ignore'))
        except Exception:
            return False

    def find_matching_label_files(self, log_file):
        """Find matching label file for a log file."""
        log_name = log_file.stem
        possible_matches = []
        
        # Use recursive glob to search all subdirectories in labels_dir
        for file in self.labels_dir.rglob(f"{log_name}*"):
            if self.is_text_file(file):
                possible_matches.append(file)
                
        if not possible_matches:
            logger.warning(f"No matching label file found for {log_file}")
            return None
            
        if len(possible_matches) > 1:
            for match in possible_matches:
                if match.suffix == log_file.suffix:
                    return match
            log_ext = log_file.suffix
            if log_ext and log_ext[1:].isdigit():
                for match in possible_matches:
                    if match.suffix and match.suffix[1:].isdigit():
                        return match
            logger.info(f"Multiple label candidates for {log_file}, using {possible_matches[0]}")
            
        return possible_matches[0]

    def read_file_lines(self, file_path):
        """Read lines from a file, handling encoding issues gracefully."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return [line.rstrip('\n') for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return []

    def read_label_map(self, label_file):
        """Read label mappings from a label file."""
        if not label_file:
            return {}
        label_map = {}
        try:
            with open(label_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if 'line' in item and 'labels' in item:
                            label_map[item['line']] = item['labels']
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line in {label_file}")
            return label_map
        except Exception as e:
            logger.error(f"Error reading label file {label_file}: {str(e)}")
            return {}

    def serialize_example(self, log, labels):
        """Create a TensorFlow Example for serialization."""
        feature = {
            'l': tf.train.Feature(bytes_list=tf.train.BytesList(value=[log.encode('utf-8')])),
            'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[json.dumps(labels).encode('utf-8')])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    def process_file(self, log_file):
        """Process a log file with its matching label file."""
        log_file = Path(log_file)
        
        if not self.is_text_file(log_file):
            logger.info(f"Skipping non-text file: {log_file}")
            return
            
        logger.info(f"Processing {log_file}")

        label_file = self.find_matching_label_files(log_file)
        label_map = self.read_label_map(label_file) if label_file else {}
        log_lines = self.read_file_lines(log_file)
        
        if not log_lines:
            logger.warning(f"No text content found in {log_file}")
            return

        # Preserve directory structure by getting relative path from logs_dir
        rel_path = log_file.relative_to(self.logs_dir)
        # Use parent directories as part of the output filename
        if rel_path.parent != Path('.'):
            # Create dataset/category based output path
            output_path = self.output_dir / f"{rel_path.parent.as_posix().replace('/', '_')}_{log_file.stem}.tfrecord"
        else:
            output_path = self.output_dir / f"{log_file.stem}.tfrecord"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tf.io.TFRecordWriter(
            str(output_path), 
            options=tf.io.TFRecordOptions(compression_type="GZIP")
        ) as writer:
            for idx, line in enumerate(log_lines, start=1):
                labels = label_map.get(idx, [])
                example = self.serialize_example(line, labels)
                writer.write(example)

        logger.info(f"Wrote {len(log_lines)} records to {output_path}")

    def batch_process(self):
        """Process all valid log files in the logs directory."""
        log_files = list(self.logs_dir.rglob('*'))
        log_files = [f for f in log_files if f.is_file() and not f.name.startswith('.')]

        logger.info(f"Found {len(log_files)} potential log files")
        processed_count = 0
        
        for log_file in log_files:
            if self.is_text_file(log_file):
                self.process_file(log_file)
                processed_count += 1
                
        logger.info(f"Batch processing complete. Processed {processed_count} text files.")

def main():
    preprocessor = LogPreprocessor()
    preprocessor.batch_process()

if __name__ == '__main__':
    main()
