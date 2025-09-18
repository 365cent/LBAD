import json
import logging
import os
import mimetypes
from bisect import bisect_left
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure Matplotlib can write cache/config on HPC
try:
    if "MPLCONFIGDIR" not in os.environ:
        _mpl_dir = Path.cwd() / ".mplconfig"
        _mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_dir)
except Exception:
    pass

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

        # Hot path caches to avoid repeated disk access and expensive recomputation
        self._text_check_cache: Dict[Path, Tuple[Tuple[int, int], bool]] = {}
        self._log_type_cache: Dict[str, str] = {}
        self._label_lookup_cache: Dict[Tuple[str, str], Optional[Path]] = {}
        self._label_map_cache: Dict[Path, Tuple[Tuple[int, int], Dict[int, List[Any]]]] = {}
        self._label_names, self._label_paths = self._build_label_index()
        
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
        file_path = Path(file_path)

        try:
            stat_result = file_path.stat()
        except (OSError, ValueError):
            return False

        signature = (stat_result.st_size, stat_result.st_mtime_ns)
        cached = self._text_check_cache.get(file_path)
        if cached and cached[0] == signature:
            return cached[1]

        # First check MIME type using extension hint
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type.startswith('text/'):
            result = True
        else:
            try:
                with open(file_path, 'rb') as handle:
                    sample = handle.read(1024)  # Sample first 1KB
                    if b'\0' in sample:
                        result = False
                    else:
                        try:
                            sample.decode('utf-8')
                            result = True
                        except UnicodeDecodeError:
                            result = bool(sample.decode('latin-1', errors='ignore'))
            except Exception:
                result = False

        self._text_check_cache[file_path] = (signature, result)
        return result

    def _build_label_index(self) -> Tuple[List[str], List[Path]]:
        """Pre-index label files by name for fast prefix lookup."""
        if not self.labels_dir.exists():
            return [], []

        candidates: List[Tuple[str, Path]] = []
        for label_path in self.labels_dir.rglob('*'):
            if not label_path.is_file() or label_path.name.startswith('.'):
                continue
            candidates.append((label_path.name.lower(), label_path))

        if not candidates:
            return [], []

        candidates.sort(key=lambda item: item[0])
        names, paths = zip(*candidates)
        return list(names), list(paths)

    def determine_log_type(self, log_file):
        """Determine the log type based on the path and filename."""
        log_file = Path(log_file)
        cache_key = str(log_file)
        cached = self._log_type_cache.get(cache_key)
        if cached is not None:
            return cached

        path_str = cache_key.lower()
        filename = log_file.name.lower()
        
        # Use directory structure to determine log type
        if '/vpn/logs/' in path_str and 'openvpn.log' in filename:
            self._log_type_cache[cache_key] = 'vpn'
            return 'vpn'

        if '/inet-firewall/logs/' in path_str and 'dnsmasq.log' in filename:
            self._log_type_cache[cache_key] = 'dns'
            return 'dns'
            
        if '/internal_share/logs/audit/' in path_str and 'audit.log' in filename:
            self._log_type_cache[cache_key] = 'share'
            return 'share'
            
        if '/intranet_server/logs/' in path_str:
            if '/apache2/' in path_str:
                if 'access.log' in filename:
                    self._log_type_cache[cache_key] = 'wp-access'
                    return 'wp-access'
                elif 'error.log' in filename:
                    self._log_type_cache[cache_key] = 'wp-error'
                    return 'wp-error'
            elif '/auth.log' in path_str:
                self._log_type_cache[cache_key] = 'auth'
                return 'auth'
            elif '/audit/audit.log' in path_str:
                self._log_type_cache[cache_key] = 'audit'
                return 'audit'
            elif 'error.log' in filename:
                self._log_type_cache[cache_key] = 'intranet-error'
                return 'intranet-error'
        
        # Check for monitoring logs (CPU logs)
        if '/monitoring/logs/' in path_str and 'system.cpu.log' in filename:
            self._log_type_cache[cache_key] = 'monitor'
            return 'monitor'
                
        # Fallback to more generic checks based on file patterns
        for log_type, patterns in self.log_types.items():
            for pattern in patterns:
                if pattern.lower() in filename:
                    # For audit logs, disambiguate between share and regular audit
                    if log_type == 'audit' and 'audit.log' in filename:
                        if '/internal_share/' in path_str:
                            self._log_type_cache[cache_key] = 'share'
                            return 'share'
                        else:
                            self._log_type_cache[cache_key] = 'audit'
                            return 'audit'
                    # For error logs, disambiguate between wp-error and intranet-error
                    elif log_type == 'wp-error' and 'error.log' in filename:
                        if '/apache2/' in path_str:
                            self._log_type_cache[cache_key] = 'wp-error'
                            return 'wp-error'
                        else:
                            self._log_type_cache[cache_key] = 'intranet-error'
                            return 'intranet-error'
                    # For CPU logs
                    elif 'cpu.log' in filename or 'system.cpu' in filename:
                        self._log_type_cache[cache_key] = 'monitor'
                        return 'monitor'
                    self._log_type_cache[cache_key] = log_type
                    return log_type
        
        # Default to unknown if no match found
        logger.warning(f"Could not determine log type for {log_file}, categorizing as unknown")
        self._log_type_cache[cache_key] = 'unknown'
        return 'unknown'

    def find_matching_label_files(self, log_file):
        """Find matching label file for a log file."""
        log_file = Path(log_file)
        log_name = log_file.stem
        cache_key = (log_name, log_file.suffix.lower())
        if cache_key in self._label_lookup_cache:
            return self._label_lookup_cache[cache_key]

        candidates = self._collect_label_candidates(log_name)

        if not candidates:
            logger.warning(f"No matching label file found for {log_file}")
            self._label_lookup_cache[cache_key] = None
            return None

        if len(candidates) == 1:
            selected = candidates[0]
        else:
            selected = self._select_preferred_label(candidates, log_file)
            if selected is None:
                selected = candidates[0]
            if len(candidates) > 1:
                logger.info(f"Multiple label candidates for {log_file}, using {selected}")

        self._label_lookup_cache[cache_key] = selected
        return selected

    def _collect_label_candidates(self, log_name: str) -> List[Path]:
        if not self._label_names:
            return []

        lower_prefix = log_name.lower()
        idx = bisect_left(self._label_names, lower_prefix)
        matches: List[Path] = []

        while idx < len(self._label_names):
            candidate_name = self._label_names[idx]
            if not candidate_name.startswith(lower_prefix):
                break
            candidate_path = self._label_paths[idx]
            if self.is_text_file(candidate_path):
                matches.append(candidate_path)
            idx += 1

        return matches

    @staticmethod
    def _select_preferred_label(candidates: List[Path], log_file: Path) -> Optional[Path]:
        exact_suffix = next((c for c in candidates if c.suffix == log_file.suffix), None)
        if exact_suffix is not None:
            return exact_suffix

        log_ext = log_file.suffix
        if log_ext and log_ext[1:].isdigit():
            numeric_match = next((c for c in candidates if c.suffix and c.suffix[1:].isdigit()), None)
            if numeric_match is not None:
                return numeric_match

        return None

    def read_label_map(self, label_file):
        """Read label mappings from a label file."""
        if not label_file:
            return {}
        label_file = Path(label_file)

        try:
            stat_result = label_file.stat()
        except (OSError, ValueError):
            logger.error(f"Error reading label file {label_file}: cannot stat path")
            return {}

        signature = (stat_result.st_size, stat_result.st_mtime_ns)
        cached = self._label_map_cache.get(label_file)
        if cached and cached[0] == signature:
            return cached[1]

        label_map: Dict[int, List[Any]] = {}
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
        except Exception as e:
            logger.error(f"Error reading label file {label_file}: {str(e)}")
            return {}

        self._label_map_cache[label_file] = (signature, label_map)
        return label_map

    def serialize_example(self, log, labels, log_type):
        """Create a TensorFlow Example for serialization."""
        feature = {
            'l': tf.train.Feature(bytes_list=tf.train.BytesList(value=[log.encode('utf-8')])),
            'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[json.dumps(labels).encode('utf-8')])),
            'log_type': tf.train.Feature(bytes_list=tf.train.BytesList(value=[log_type.encode('utf-8')])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    def process_file(self, log_file, *, skip_text_check: bool = False):
        """Process a log file with its matching label file.
        Returns (processed: bool, log_type: str | None)
        """
        log_file = Path(log_file)

        if not skip_text_check and not self.is_text_file(log_file):
            logger.info(f"Skipping non-text file: {log_file}")
            return False, None
            
        # Determine log type early to compute output path
        log_type = self.determine_log_type(log_file)
        logger.info(f"Processing {log_file} (type={log_type})")

        # Prepare output path and skip if already exists
        type_output_dir = self.output_dir / log_type
        type_output_dir.mkdir(parents=True, exist_ok=True)

        # Preserve directory structure by getting relative path from logs_dir
        try:
            rel_path = log_file.relative_to(self.logs_dir)
        except ValueError:
            rel_path = Path(log_file.name)

        # Create output path that includes user directory and log name
        if rel_path.parent != Path('.') and rel_path.parts:
            # Extract the username (first directory in path)
            user = rel_path.parts[0]
            output_path = type_output_dir / f"{user}_{log_file.stem}.tfrecord"
        else:
            output_path = type_output_dir / f"{log_file.stem}.tfrecord"

        try:
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Output already exists, skipping: {output_path}")
                return False, log_type
        except OSError:
            # If stat fails but file exists, attempt to overwrite
            pass

        # Load inputs only if we are going to write
        label_file = self.find_matching_label_files(log_file)
        label_map = self.read_label_map(label_file) if label_file else {}
        records_written = 0

        with ExitStack() as stack:
            try:
                reader = stack.enter_context(log_file.open('r', encoding='utf-8', errors='replace'))
            except Exception as exc:
                logger.error(f"Error reading {log_file}: {exc}")
                return False, log_type

            writer = None

            for line in reader:
                cleaned = line.rstrip('\n')
                if not cleaned.strip():
                    continue
                records_written += 1
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = stack.enter_context(
                        tf.io.TFRecordWriter(
                            str(output_path),
                            options=tf.io.TFRecordOptions(compression_type="GZIP")
                        )
                    )
                labels = label_map.get(records_written, [])
                example = self.serialize_example(cleaned, labels, log_type)
                writer.write(example)

        if records_written == 0:
            logger.warning(f"No text content found in {log_file}")
            return False, log_type

        logger.info(f"Wrote {records_written} records to {output_path} with log type {log_type}")
        return True, log_type

    def batch_process(self):
        """Process all valid log files in the logs directory."""
        if not self.logs_dir.exists():
            logger.warning(f"Logs directory not found: {self.logs_dir}")
            return

        log_files = (
            path for path in self.logs_dir.rglob('*')
            if path.is_file() and not path.name.startswith('.')
        )

        potential_count = 0
        processed_count = 0
        log_type_counts: Dict[str, int] = {}

        for log_file in log_files:
            potential_count += 1
            if not self.is_text_file(log_file):
                continue
            processed, log_type = self.process_file(log_file, skip_text_check=True)
            if processed:
                processed_count += 1
                resolved_type = log_type or self.determine_log_type(log_file)
                log_type_counts[resolved_type] = log_type_counts.get(resolved_type, 0) + 1

        logger.info(f"Found {potential_count} potential log files")
        logger.info(f"Batch processing complete. Processed {processed_count} text files.")
        logger.info(f"Log type distribution: {log_type_counts}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess log files into TFRecord format")
    parser.add_argument("--log-type", type=str, default=None, help="Process only this specific log type")
    args = parser.parse_args()
    
    preprocessor = LogPreprocessor()
    
    if args.log_type:
        # Process specific log type if requested
        print(f"Processing specific log type: {args.log_type}")
        # Find files for this log type and process them
        processed_count = 0
        candidate_files = (
            path for path in preprocessor.logs_dir.rglob('*')
            if path.is_file() and not path.name.startswith('.')
        )

        for log_file in candidate_files:
            if not preprocessor.is_text_file(log_file):
                continue
            determined_type = preprocessor.determine_log_type(log_file)
            if determined_type == args.log_type:
                processed, _ = preprocessor.process_file(log_file, skip_text_check=True)
                if processed:
                    processed_count += 1
        
        print(f"Processed {processed_count} files for log type '{args.log_type}'")
    else:
        # Process all log types
        preprocessor.batch_process()

if __name__ == '__main__':
    main()
