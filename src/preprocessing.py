import json
import logging
import os
import mimetypes
from pathlib import Path
import tensorflow as tf
import re

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
        
        # Define log types based on the table
        self.log_types = {
            'vpn': ['openvpn.log'],
            'wp-access': ['access.log'],
            'wp-error': ['error.log'],
            'auth': ['auth.log'],
            'audit': ['audit.log'],
            'dns': ['dnsmasq.log'],
            'monitor': ['system.cpu.log']
        }

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

    def determine_log_type(self, log_file):
        """Determine the log type based on the path and filename."""
        path_str = str(log_file).lower()
        filename = log_file.name.lower()
        parts = Path(path_str).parts
        
        # Use directory structure to determine log type
        if '/vpn/logs/' in path_str and 'openvpn.log' in filename:
            return 'vpn'
        
        if '/inet-firewall/logs/' in path_str and 'dnsmasq.log' in filename:
            return 'dns'
            
        if '/internal_share/logs/audit/' in path_str and 'audit.log' in filename:
            return 'share'
            
        if '/intranet_server/logs/' in path_str:
            if '/apache2/' in path_str:
                if 'access.log' in filename:
                    return 'wp-access'
                elif 'error.log' in filename:
                    return 'wp-error'
            elif '/auth.log' in path_str:
                return 'auth'
            elif '/audit/audit.log' in path_str:
                return 'audit'
            elif 'error.log' in filename:
                return 'intranet-error'
        
        # Check for monitoring logs (CPU logs)
        if '/monitoring/logs/' in path_str and 'system.cpu.log' in filename:
            return 'monitor'
                
        # Fallback to more generic checks based on file patterns
        for log_type, patterns in self.log_types.items():
            for pattern in patterns:
                if pattern.lower() in filename:
                    # For audit logs, disambiguate between share and regular audit
                    if log_type == 'audit' and 'audit.log' in filename:
                        if '/internal_share/' in path_str:
                            return 'share'
                        else:
                            return 'audit'
                    # For error logs, disambiguate between wp-error and intranet-error
                    elif log_type == 'wp-error' and 'error.log' in filename:
                        if '/apache2/' in path_str:
                            return 'wp-error'
                        else:
                            return 'intranet-error'
                    # For CPU logs
                    elif 'cpu.log' in filename or 'system.cpu' in filename:
                        return 'monitor'
                    return log_type
        
        # Default to unknown if no match found
        logger.warning(f"Could not determine log type for {log_file}, categorizing as unknown")
        return 'unknown'

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

    def serialize_example(self, log, labels, log_type):
        """Create a TensorFlow Example for serialization."""
        feature = {
            'l': tf.train.Feature(bytes_list=tf.train.BytesList(value=[log.encode('utf-8')])),
            'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[json.dumps(labels).encode('utf-8')])),
            'log_type': tf.train.Feature(bytes_list=tf.train.BytesList(value=[log_type.encode('utf-8')])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    def process_file(self, log_file):
        """Process a log file with its matching label file.
        Returns (processed: bool, log_type: str | None)
        """
        log_file = Path(log_file)
        
        if not self.is_text_file(log_file):
            logger.info(f"Skipping non-text file: {log_file}")
            return False, None
            
        # Determine log type early to compute output path
        log_type = self.determine_log_type(log_file)
        logger.info(f"Processing {log_file} (type={log_type})")

        # Prepare output path and skip if already exists
        type_output_dir = self.output_dir / log_type
        type_output_dir.mkdir(parents=True, exist_ok=True)

        # Preserve directory structure by getting relative path from logs_dir
        rel_path = log_file.relative_to(self.logs_dir)

        # Create output path that includes user directory and log name
        if rel_path.parent != Path('.'):
            # Extract the username (first directory in path)
            user = rel_path.parts[0] if rel_path.parts else "unknown"
            output_path = type_output_dir / f"{user}_{log_file.stem}.tfrecord"
        else:
            output_path = type_output_dir / f"{log_file.stem}.tfrecord"

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Output already exists, skipping: {output_path}")
            return False, log_type

        # Load inputs only if we are going to write
        label_file = self.find_matching_label_files(log_file)
        label_map = self.read_label_map(label_file) if label_file else {}
        log_lines = self.read_file_lines(log_file)
        
        if not log_lines:
            logger.warning(f"No text content found in {log_file}")
            return False, log_type
        
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tf.io.TFRecordWriter(
            str(output_path), 
            options=tf.io.TFRecordOptions(compression_type="GZIP")
        ) as writer:
            for idx, line in enumerate(log_lines, start=1):
                labels = label_map.get(idx, [])
                example = self.serialize_example(line, labels, log_type)
                writer.write(example)

        logger.info(f"Wrote {len(log_lines)} records to {output_path} with log type {log_type}")
        return True, log_type

    def batch_process(self):
        """Process all valid log files in the logs directory."""
        log_files = list(self.logs_dir.rglob('*'))
        log_files = [f for f in log_files if f.is_file() and not f.name.startswith('.')]

        logger.info(f"Found {len(log_files)} potential log files")
        processed_count = 0
        log_type_counts = {}
        
        for log_file in log_files:
            if self.is_text_file(log_file):
                processed, log_type = self.process_file(log_file)
                if processed:
                    processed_count += 1
                    # Track counts by log type
                    if log_type is None:
                        log_type = self.determine_log_type(log_file)
                    log_type_counts[log_type] = log_type_counts.get(log_type, 0) + 1
                
        logger.info(f"Batch processing complete. Processed {processed_count} text files.")
        logger.info(f"Log type distribution: {log_type_counts}")

def main():
    preprocessor = LogPreprocessor()
    preprocessor.batch_process()

if __name__ == '__main__':
    main()