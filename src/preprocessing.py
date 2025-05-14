import json
import logging
import os
import mimetypes
import re
import argparse
from pathlib import Path
import tensorflow as tf
from typing import Any, Dict, List, Set, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LogPreprocessor")

# Initialize mimetypes database
mimetypes.init()

# Default directory names
_DEFAULT_LOGS_DIR_NAME = "logs"
_DEFAULT_LABELS_DIR_NAME = "labels"
_DEFAULT_PROCESSED_DIR_NAME = "processed"

class LogPreprocessor:
    """
    Preprocesses log files, detects their types, associates them with labels,
    and serializes them into TFRecord format.
    """

    # Mappings from cluster labels to primitive activities
    _CLUSTER_TO_PRIMITIVE_MAP: Dict[str, List[str]] = {
        'escalate': [
            'escalated_command', 'escalated_sudo_session',
            'attacker_change_user', 'reverse_shell', 'attacker_vpn'
        ],
        'dnsteal': ['dnsteal-received', 'dnsteal-dropped'],
        'foothold': ['webshell_cmd', 'webshell_upload', 'dirb'],
        'exfiltration_service': ['dnsteal-received', 'dnsteal-dropped'],
        'attacker_http': ['webshell_cmd', 'webshell_upload', 'dirb'],
        'escalated_command': [
            'escalated_sudo_session', 'attacker_change_user'
        ]
    }

    # Extended log type detection patterns
    _LOG_PATTERNS: Dict[str, List[str]] = {
        "web": [
            r"GET /", r"POST /", r"HTTP/\d\.\d",
            r"^\S+ \S+ \S+ \[\d+/\w+/\d+:\d+:\d+:\d+ [+-]\d+\] "
            r"\"(?:GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)",
            r"Apache/\d", r"nginx/\d",
            r"Mozilla/\d", r"User-Agent:", r"Referer:",
            # Added patterns for common web attacks
            r"(?:%3C|<)script(?:%3E|>)", r"javascript:", r"onmouseover=", # XSS
            r"select.+from", r"union.+select", r"drop.+table", # SQL Injection
            r"\\.\\./", # Directory Traversal
        ],
        "error": [
            r"\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \w{3} \d{2} \d{2}:\d{2}:\d{2}.\d+ \d{4}\]",
            r"\[(?:authz_core|php\d*|cgi|negotiation|core|mpm_event|ssl|"
            r"access_compat):error\]",
            r"AH\d{5}:",
            r"client denied by server configuration",
            r"script '[^\']+' not found or unable to stat",
            r"File does not exist:",
            r"Invalid URI in request",
            r"Negotiation: discovered file\(s\) matching request",
            # Added patterns for errors related to attacks
            r"SQL syntax error", r"failed login", r"permission denied"
        ],
        "auth": [
            r"sshd", r"authentication", r"session opened", r"invalid user",
            r"Failed password", r"Accepted password", r"pam_unix",
            r"login:", r"su:", r"sudo:", r"user NOT in sudoers",
            r"authentication failure", r"auth\.log", r"FAILED LOGIN",
            r"session closed", r"password changed", r"new user", r"user added",
            # Added patterns for brute force
            r"Too many authentication failures", r"maximum authentication attempts exceeded"
        ],
        "dns": [
            r"query\[A\]", r"query\[AAAA\]", r"named\[\d+\]", r"dnsmasq",
            r"IN A", r"IN AAAA", r"IN NS", r"IN MX", r"IN TXT", r"IN SOA",
            r"bind", r"DNS resolution", r"domain lookup", r"resolv\.conf",
            r"NXDOMAIN", r"SERVFAIL", r"NOERROR", r"REFUSED"
        ],
        "firewall": [
            r"ACCEPT", r"DROP", r"REJECT", r"ALLOWED", r"DENIED",
            r"SRC=", r"DST=",
            r"iptables", r"firewalld", r"ufw",
            r"FORWARD", r"INPUT", r"OUTPUT",
            r"SPT=\d+", r"DPT=\d+", r"PROTO=\w+", r"MAC=", r"LEN=\d+",
            r"IN=\w+", r"OUT=\w+", r"TTL=\d+", r"packet filtered"
        ],
        "vpn": [
            r"openvpn", r"wireguard", r"tunnel", r"tun\d+", r"tap\d+",
            r"VPN", r"connected", r"disconnected", r"handshake",
            r"tls_error", r"auth failed", r"peer connection",
            r"key exchange", r"IPsec", r"L2TP", r"PPTP",
            r"connection established"
        ],
        "audit": [
            r"audit\[\d+\]", r"AVC", r"apparmor", r"type=\w+", r"audit:",
            r"selinux", r"SYSCALL", r"USER_CMD", r"USER_AUTH", r"USER_ACCT",
            r"cred_disp", r"cred_acq", r"auid=\d+", r"ses=\d+", r"subj=",
            r"msg='op=", r"audispd", r"auditd"
        ],
        "kernel": [
            r"kernel: \[.*\]", r"sysctl", r"dmesg", r"module loaded",
            r"CPU\d+", r"Memory:", r"I/O", r"IRQ", r"ACPI", r"PCI", r"USB",
            r"ALSA", r"BIOS", r"thermal", r"OOM killer"
        ],
        "systemd": [
            r"systemd\[\d+\]", r"Starting", r"Stopped", r"Service",
            r"Unit", r"Process", r"Job", r"failed", r"succeeded",
            r"Started", r"Reloading", r"Mounting", r"Unmounting", r"journald"
        ],
        "ids": [ # Added new IDS log type
            r"Snort", r"Suricata", r"ET POLICY", r"ET MALWARE", r"GPL ATTACK_RESPONSE",
            r"\\[\\*\\*\\] \\[\\d+:\\d+:\\d+\\]", # Common Snort/Suricata alert prefix
            r"Suspicious activity", r"Potential malware", r"Command and Control",
            r"Portscan detected", r"SQL Injection attempt", r"XSS attempt", r"DDoS activity"
        ]
    }

    def __init__(
        self,
        logs_dir: Optional[Union[str, Path]] = None,
        labels_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        target_log_type: Optional[str] = None
    ):
        """Initializes the LogPreprocessor.

        Args:
            logs_dir: Directory containing raw log files.
                      Defaults to 'logs' in the project root.
            labels_dir: Directory containing label files.
                        Defaults to 'labels' in the project root.
            output_dir: Directory to save processed TFRecord files.
                        Defaults to 'processed' in the project root.
            target_log_type: If specified, only process logs of this type.
        """
        base_dir = Path(__file__).resolve().parent.parent

        self.logs_dir = (base_dir / _DEFAULT_LOGS_DIR_NAME
                         if logs_dir is None else Path(logs_dir))
        self.labels_dir = (base_dir / _DEFAULT_LABELS_DIR_NAME
                           if labels_dir is None else Path(labels_dir))
        self.output_dir = (base_dir / _DEFAULT_PROCESSED_DIR_NAME
                           if output_dir is None else Path(output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_log_type = target_log_type

        if self.target_log_type:
            logger.info(
                f"Processing only log type: {self.target_log_type}"
            )
            if self.target_log_type not in self._LOG_PATTERNS:
                logger.warning(
                    f"Unknown target log type: {self.target_log_type}"
                )
                logger.info(
                    f"Available log types: {', '.join(self._LOG_PATTERNS.keys())}"
                )

    def is_text_file(self, file_path: Path) -> bool:
        """Determines if a file is likely a text file.

        Checks MIME type first, then samples content for null bytes or
        decoding issues.

        Args:
            file_path: The path to the file.

        Returns:
            True if the file is likely text, False otherwise.
        """
        mime_type, _ = mimetypes.guess_type(file_path.as_uri())
        if mime_type and mime_type.startswith('text/'):
            return True

        try:
            with file_path.open('rb') as f:
                sample = f.read(1024)  # Sample first 1KB
            if b'\0' in sample: # Check for null bytes
                return False
            # Try decoding as UTF-8, then as Latin-1 as a fallback
            sample.decode('utf-8')
            return True
        except UnicodeDecodeError:
            try:
                sample.decode('latin-1') # Common fallback
                return True
            except UnicodeDecodeError:
                return False # Still can't decode
        except OSError: # Could be FileNotFoundError, PermissionError, etc.
            return False
        # return False # This line is unreachable due to previous returns or exceptions

    def _find_matching_label_file_heuristic(
        self,
        log_file: Path,
        possible_matches: List[Path]
    ) -> Path:
        """Heuristic to select the best label file from multiple candidates."""
        # Prefer exact suffix match
        for match in possible_matches:
            if match.suffix == log_file.suffix:
                return match

        # Prefer numeric suffix match if log file has one (e.g., .log.1)
        log_ext_num_part = log_file.suffix[1:]
        if log_file.suffix and log_ext_num_part.isdigit():
            for match in possible_matches:
                match_ext_num_part = match.suffix[1:]
                if match.suffix and match_ext_num_part.isdigit():
                    # Could add more specific matching here if needed, e.g. exact number
                    return match
        
        # Fallback to the first match if specific heuristics don't apply
        logger.info(
            f"Multiple label candidates for {log_file.name}. "
            f"Using heuristic match: {possible_matches[0].name}"
        )
        return possible_matches[0]

    def find_matching_label_file(self, log_file: Path) -> Optional[Path]:
        """Finds a matching label file for a given log file.

        Args:
            log_file: The path to the log file.

        Returns:
            The path to the best matching label file, or None if no suitable
            label file is found.
        """
        log_name_stem = log_file.stem
        
        # Search recursively in labels_dir for files starting with log_name_stem
        candidate_label_files = [
            f for f in self.labels_dir.rglob(f"{log_name_stem}*")
            if f.is_file() and self.is_text_file(f)
        ]

        if not candidate_label_files:
            logger.warning(f"No text-based label file found for {log_file.name}")
            return None
        
        if len(candidate_label_files) == 1:
            return candidate_label_files[0]

        return self._find_matching_label_file_heuristic(
            log_file, candidate_label_files
        )

    def read_file_lines(self, file_path: Path) -> List[str]:
        """Reads lines from a file, stripping whitespace and handling errors.

        Args:
            file_path: The path to the file.

        Returns:
            A list of non-empty lines from the file.
        """
        try:
            with file_path.open('r', encoding='utf-8', errors='replace') as f:
                return [line.rstrip('\n') for line in f if line.strip()]
        except OSError as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []

    def read_label_map(
        self,
        label_file: Path
    ) -> Dict[int, List[str]]:
        """Reads label mappings from a JSON-lines label file.

        Args:
            label_file: Path to the label file. Each line should be a JSON
                        object with 'line' (int) and 'labels' (list of str).

        Returns:
            A dictionary mapping line numbers to a list of labels.
        """
        label_map: Dict[int, List[str]] = {}
        try:
            with label_file.open('r', encoding='utf-8', errors='replace') as f:
                for line_number, content in enumerate(f, 1):
                    content = content.strip()
                    if not content:
                        continue
                    try:
                        item = json.loads(content)
                        # Ensure 'line' is an int and 'labels' is a list of strings
                        if (isinstance(item.get('line'), int) and
                            isinstance(item.get('labels'), list) and
                            all(isinstance(lbl, str) for lbl in item['labels'])):
                            label_map[item['line']] = item['labels']
                        else:
                            logger.warning(
                                f"Skipping malformed or incorrectly typed data in "
                                f"{label_file.name}, line {line_number}: {content}"
                            )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Skipping malformed JSON in {label_file.name}, "
                            f"line {line_number}: {content}"
                        )
            return label_map
        except OSError as e:
            logger.error(f"Error reading label file {label_file}: {e}")
            return {}

    def expand_cluster_labels(self, labels: List[str]) -> List[str]:
        """Converts cluster labels to their primitive activity components.

        Args:
            labels: A list of labels, which may include cluster labels.

        Returns:
            A list of primitive activity labels, with duplicates removed.
        """
        expanded_labels: Set[str] = set()
        for label in labels:
            if label in self._CLUSTER_TO_PRIMITIVE_MAP:
                expanded_labels.update(self._CLUSTER_TO_PRIMITIVE_MAP[label])
            else:
                expanded_labels.add(label)
        return sorted(list(expanded_labels)) # Sort for consistent output

    def detect_log_type(self, log_lines: List[str]) -> str:
        """Detects the log type based on content patterns.

        Args:
            log_lines: A list of log lines (typically the first few lines
                       of a log file).

        Returns:
            The detected log type (e.g., 'web', 'error') or 'unknown'.
        """
        if not log_lines:
            return "unknown"

        # Sample up to the first 20 lines for pattern matching
        sample_text = "\n".join(log_lines[:20])
        
        type_matches: Dict[str, int] = {}
        for log_type_key, patterns in self._LOG_PATTERNS.items():
            # Count how many patterns for this log_type_key match the sample_text
            num_matches = sum(
                1 for pattern in patterns
                if re.search(pattern, sample_text, re.IGNORECASE)
            )
            if num_matches > 0:
                type_matches[log_type_key] = num_matches
        
        if not type_matches:
            return "unknown"

        # Heuristic: Prioritize 'error' logs if they have significant matches
        if 'error' in type_matches and type_matches['error'] > 1:
            if 'web' in type_matches:
                # If error matches are strong relative to web matches, prefer error
                if type_matches['error'] >= type_matches['web'] * 0.8:
                    return 'error'
            else: # Only error matches significantly (and no web matches)
                return 'error'
            
        # Default to the log type with the most pattern matches
        return max(type_matches.items(), key=lambda item: item[1])[0]

    def _serialize_example(
        self, log_line: str, labels: List[str]
    ) -> str:
        """Creates a TensorFlow Example for serialization.

        Labels are expanded to primitive activities before serialization.

        Args:
            log_line: A single log line string.
            labels: A list of labels associated with the log line.

        Returns:
            A serialized tf.train.Example string.
        """
        primitive_labels = self.expand_cluster_labels(labels)
        feature = {
            'l': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[log_line.encode('utf-8')])
            ),
            'y': tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[json.dumps(primitive_labels).encode('utf-8')]
                )
            )
        }
        return tf.train.Example(
            features=tf.train.Features(feature=feature)
        ).SerializeToString()

    def process_file(self, log_file_path: Path) -> None:
        """Processes a single log file and saves it as a TFRecord file.

        Args:
            log_file_path: The path to the log file to process.
        """
        if not self.is_text_file(log_file_path):
            logger.info(f"Skipping non-text file: {log_file_path.name}")
            return

        logger.info(f"Processing {log_file_path.name}")
        log_lines = self.read_file_lines(log_file_path)
        if not log_lines:
            logger.warning(
                f"No text content found in {log_file_path.name}, skipping."
            )
            return

        detected_log_type = self.detect_log_type(log_lines)
        if self.target_log_type and detected_log_type != self.target_log_type:
            logger.info(
                f"Skipping {log_file_path.name} (detected type: "
                f"{detected_log_type}, target: {self.target_log_type})"
            )
            return
        
        logger.info(
            f"Identified {log_file_path.name} as '{detected_log_type}' log."
        )

        label_file_path = self.find_matching_label_file(log_file_path)
        label_map: Dict[int, List[str]] = {}
        if label_file_path:
            logger.info(f"Using label file: {label_file_path.name}")
            label_map = self.read_label_map(label_file_path)
        else:
            logger.info(f"No label file found for {log_file_path.name}.")

        # Prepare output directory and file path
        log_type_output_dir = self.output_dir / detected_log_type
        log_type_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Try to get relative path from the main logs_dir
            # to preserve parts of the original structure in the filename.
            rel_path = log_file_path.relative_to(self.logs_dir)
            if rel_path.parent != Path('.'): # If it was in a subdirectory
                output_filename_stem = (
                    f"{rel_path.parent.as_posix().replace('/', '_')}_"
                    f"{log_file_path.stem}"
                )
            else:
                output_filename_stem = log_file_path.stem
        except ValueError: # If log_file_path is not under self.logs_dir
            output_filename_stem = log_file_path.stem


        output_tfrecord_path = (
            log_type_output_dir / f"{output_filename_stem}.tfrecord"
        )
        
        try:
            with tf.io.TFRecordWriter(
                output_tfrecord_path.as_posix(),
                options=tf.io.TFRecordOptions(compression_type="GZIP")
            ) as writer:
                for line_num, log_line_content in enumerate(log_lines, start=1):
                    # Labels are 1-indexed in typical label files
                    current_labels = label_map.get(line_num, [])
                    example = self._serialize_example(
                        log_line_content, current_labels
                    )
                    writer.write(example)
            logger.info(
                f"Wrote {len(log_lines)} records to {output_tfrecord_path.name}"
            )
        except (OSError, tf.errors.OpError) as e: # Catch file write or TF errors
            logger.error(
                f"Error writing TFRecord file {output_tfrecord_path}: {e}"
            )


    def batch_process(self) -> None:
        """Processes all valid log files in the configured logs directory."""
        log_files_to_process: List[Path] = []
        for f_path in self.logs_dir.rglob('*'):
            if f_path.is_file() and not f_path.name.startswith('.'):
                if self.is_text_file(f_path): # Pre-filter for text files
                    log_files_to_process.append(f_path)
                else:
                    logger.debug(f"Skipping non-text file in batch: {f_path.name}")


        logger.info(
            f"Found {len(log_files_to_process)} potential text log files "
            f"in {self.logs_dir}"
        )
        
        processed_count = 0
        log_type_counts: Dict[str, int] = {}

        for log_file_path in log_files_to_process:
            # If target_log_type is set, we do a quick detection first
            # to avoid full processing if it won't match.
            if self.target_log_type:
                # Read minimal lines for type detection
                temp_lines = self.read_file_lines(log_file_path)
                if not temp_lines: continue # Skip if file became empty or unreadable
                
                detected_type = self.detect_log_type(temp_lines)
                if detected_type != self.target_log_type:
                    logger.debug(
                        f"Skipping {log_file_path.name} (type {detected_type}) "
                        f"due to target type {self.target_log_type}."
                    )
                    continue
            
            self.process_file(log_file_path) # This will re-detect type, but ensures consistency
            
            # For accurate counting after processing (which determines final type)
            # We need to get the actual type used. This is a bit redundant if
            # process_file stores the type, but let's re-detect for summary.
            # A better way would be for process_file to return the detected type.
            # For now, keeping it simple and re-reading a few lines.
            processed_lines = self.read_file_lines(log_file_path) # Re-read for safety
            if processed_lines:
                final_log_type = self.detect_log_type(processed_lines)
                if not self.target_log_type or final_log_type == self.target_log_type:
                    processed_count +=1
                    log_type_counts[final_log_type] = (
                        log_type_counts.get(final_log_type, 0) + 1
                    )
        
        if self.target_log_type:
            logger.info(
                f"Batch processing complete. "
                f"Processed {log_type_counts.get(self.target_log_type, 0)} "
                f"files matching target type '{self.target_log_type}'."
            )
        else:
            logger.info(
                f"Batch processing complete. Processed {processed_count} "
                f"text files."
            )

        if log_type_counts:
            logger.info("Summary of processed log types:")
            for log_type, count in sorted(log_type_counts.items()):
                logger.info(f"  {log_type}: {count} files")

            logger.info("Output directories created/used:")
            for log_type in sorted(log_type_counts.keys()):
                log_type_output_dir = self.output_dir / log_type
                if log_type_output_dir.exists(): # Check existence for clarity
                    logger.info(f"  {log_type_output_dir}")


def main() -> None:
    """Main function to parse arguments and run the preprocessor."""
    parser = argparse.ArgumentParser(
        description='Process log files and convert to TFRecord format.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--log-type',
        choices=list(LogPreprocessor._LOG_PATTERNS.keys()) + ['all'], # Added 'all'
        help='Process only logs of this specific type or all known types.'
    )
    parser.add_argument(
        '--logs-dir',
        type=Path,
        default=None, # Will use default from class
        help=f'Directory containing log files. Defaults to project_root/'
             f'{_DEFAULT_LOGS_DIR_NAME}'
    )
    parser.add_argument(
        '--labels-dir',
        type=Path,
        default=None, # Will use default from class
        help=f'Directory containing label files. Defaults to project_root/'
             f'{_DEFAULT_LABELS_DIR_NAME}'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None, # Will use default from class
        help=f'Output directory for processed TFRecord files. Defaults to '
             f'project_root/{_DEFAULT_PROCESSED_DIR_NAME}'
    )
    
    args = parser.parse_args()
    
    preprocessor = LogPreprocessor(
        logs_dir=args.logs_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        target_log_type=args.log_type
    )
    preprocessor.batch_process()

if __name__ == '__main__':
    main()
