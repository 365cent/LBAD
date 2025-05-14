import pickle
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import os
import numpy as np
from pathlib import Path # For path manipulation to infer context

# Constants for data display formatting
NUMPY_SAMPLE_ROWS = 100  # Number of rows to sample for NumPy arrays (reverted from user change for now, can be re-adjusted)
NUMPY_SAMPLE_COLS = 100  # Number of columns to sample for NumPy arrays (reverted from user change for now, can be re-adjusted)
NUMPY_DISPLAY_FULL_THRESHOLD_ITEMS = 150 # If total items < this, display full NumPy array
COLLECTION_SAMPLE_ITEMS = 15 # Number of items to sample for lists, dicts, tuples
STRING_TRUNCATE_LIMIT = 500 # Max length for displayed strings before truncation
SCROLLED_TEXT_HEIGHT = 25 # Default height for scrolled text widgets

# --- Color Palette & Font --- 
COLOR_BACKGROUND = "#F5F5F7"  # Light grey (almost white)
COLOR_TEXT_PRIMARY = "#2C3E50" # Dark blue/grey
COLOR_TEXT_SECONDARY = "#34495E" # Slightly lighter dark blue/grey
COLOR_ACCENT = "#3498DB"     # Bright blue
COLOR_ACCENT_HOVER = "#2980B9" # Darker blue for hover
COLOR_SURFACE = "#FFFFFF"     # White for text areas, card backgrounds
COLOR_BORDER = "#BDC3C7"      # Light grey for borders
FONT_FAMILY = ("Segoe UI", "Helvetica", "Arial", "sans-serif")
FONT_SIZE_NORMAL = 10
FONT_SIZE_LARGE = 12

# --- Attack Definitions (copied and adapted from fasttext_embedding.py) ---
LOG_TYPE_ATTACKS_VIEWER = {
    'dns': [
        'traceroute', 
        'dns_scan', 
        'service_scan', 
        'dnsteal-received', 
        'dnsteal-dropped'
    ],
    'network': [ # Covers firewall, vpn, and potentially generic network captures
        'traceroute', 
        'network_scan', 
        'dns_scan', 
        'service_scan',
        'port_scan', # More specific than network_scan
        'ddos', 
        'tcp_syn_flood', 
        'udp_flood', 
        'icmp_flood'
    ],
    'web': [
        'webshell_cmd', 
        'webshell_upload', 
        'dirb', 
        'wordpress_database_dump',
        'wordpress_scan',
        'sql_injection',
        'xss', # Cross-Site Scripting
        'directory_traversal',
        'command_injection' # OS Command Injection via web
    ],
    'error': [ # Errors can sometimes indicate attempted or successful attacks
        'dirb', 
        'wordpress_scan',
        'sql_injection_error', # e.g., SQL syntax error from failed attempt
        'auth_failure_reflected_in_error' # e.g., permission denied errors
    ],
    'monitoring': [ # Covers kernel, systemd - broader system activity
        'password_cracking',
        'malware_execution', # Generic malware activity detected by system monitors
        'suspicious_process'
    ],
    'auth': [
        'login_as_system_user', 
        'reverse_shell',
        'ssh_bruteforce',
        'ftp_bruteforce',
        'password_cracking' # Can also be seen here
    ],
    'audit': [
        'root_command_execution', 
        'login_as_system_user', 
        'dnsteal-received', 
        'dnsteal-dropped',
        'privilege_escalation',
        'unauthorized_file_access'
    ],
    'ids': [ # Attacks specifically flagged by IDS/IPS systems
        'port_scan_alert',
        'web_attack_alert', # Generic, could be SQLi, XSS etc.
        'malware_signature_detected',
        'ddos_alert',
        'bruteforce_alert',
        'policy_violation',
        'exploit_detected'
    ]
}

ALL_UNIQUE_ATTACKS_VIEWER = sorted(list(set(attack for attacks in LOG_TYPE_ATTACKS_VIEWER.values() for attack in attacks)))

DIR_TO_LOG_TYPE_VIEWER = {
    'dns': 'dns',
    'web': 'web',
    'auth': 'auth',
    'audit': 'audit',
    'firewall': 'network',
    'vpn': 'network',
    'kernel': 'monitoring',
    'systemd': 'monitoring',
    'error': 'error',
    'ids': 'ids',
    'snort': 'ids', 
    'suricata': 'ids' 
}

class PickleViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Embeddings and Labels Pickle Viewer")
        self.root.geometry("1100x800") # Increased size slightly
        self.root.configure(bg=COLOR_BACKGROUND)

        self.embeddings_file_path = None
        self.labels_file_path = None
        self.current_label_context_type = None # For storing 'web', 'audit', 'all_combined' etc.

        # Style
        style = ttk.Style()
        style.theme_use('clam') # 'clam' is often good for custom styling

        # General widget styling
        style.configure('.', background=COLOR_BACKGROUND, foreground=COLOR_TEXT_PRIMARY, font=(FONT_FAMILY[0], FONT_SIZE_NORMAL))

        # Frame style
        style.configure("TFrame", background=COLOR_BACKGROUND)

        # Label style
        style.configure("TLabel", background=COLOR_BACKGROUND, foreground=COLOR_TEXT_PRIMARY, font=(FONT_FAMILY[0], FONT_SIZE_NORMAL))
        style.configure("Header.TLabel", font=(FONT_FAMILY[0], FONT_SIZE_LARGE, "bold"))

        # Button style
        style.configure("TButton", 
                        background=COLOR_ACCENT, 
                        foreground=COLOR_SURFACE, 
                        font=(FONT_FAMILY[0], FONT_SIZE_NORMAL, "bold"),
                        padding=(10, 5),
                        relief="flat",
                        borderwidth=0)
        style.map("TButton",
                  background=[('active', COLOR_ACCENT_HOVER), ('pressed', COLOR_ACCENT_HOVER)],
                  relief=[('pressed', 'sunken'), ('!pressed', 'flat')])

        # LabelFrame style (may need direct configuration if ttk doesn't cover all aspects)
        style.configure("TLabelFrame", 
                        background=COLOR_SURFACE, 
                        foreground=COLOR_TEXT_PRIMARY, 
                        labelmargins=(10, 5),
                        relief="groove", 
                        borderwidth=1,
                        font=(FONT_FAMILY[0], FONT_SIZE_NORMAL, "bold"))
        # style.configure("TLabelFrame.Label",  # This line is likely causing the layout error
        #                 background=COLOR_SURFACE, 
        #                 foreground=COLOR_TEXT_PRIMARY,
        #                 font=(FONT_FAMILY[0], FONT_SIZE_NORMAL, "bold"))

        # PanedWindow style (may need direct configuration if ttk doesn't cover all aspects)
        style.configure("TPanedwindow", background=COLOR_BACKGROUND)
        # style.configure("Sash", background=COLOR_BORDER) # For PanedWindow sash, often tricky

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Controls Frame ---
        controls_frame = ttk.Frame(main_frame, style="TFrame")
        controls_frame.pack(fill=tk.X, pady=(0, 15))

        # Embeddings Selection
        embeddings_lf = ttk.LabelFrame(controls_frame, text="Embeddings File (.pkl)", padding="10")
        embeddings_lf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.open_embeddings_btn = ttk.Button(
            embeddings_lf, text="Open Embedding File", command=self.open_embeddings_file
        )
        self.open_embeddings_btn.pack(side=tk.LEFT, padx=(0,10))
        self.embeddings_file_label = ttk.Label(embeddings_lf, text="Selected: None", style="TLabel", background=COLOR_SURFACE)
        self.embeddings_file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Labels Selection
        labels_lf = ttk.LabelFrame(controls_frame, text="Labels File (.pkl)", padding="10")
        labels_lf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        self.open_labels_btn = ttk.Button(
            labels_lf, text="Open Label File", command=self.open_labels_file
        )
        self.open_labels_btn.pack(side=tk.LEFT, padx=(0,10))
        self.labels_file_label = ttk.Label(labels_lf, text="Selected: None", style="TLabel", background=COLOR_SURFACE)
        self.labels_file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Data Display PanedWindow ---
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL, style="TPanedwindow")
        paned_window.pack(fill=tk.BOTH, expand=True)

        # Embeddings Display Area
        embeddings_display_lf = ttk.LabelFrame(paned_window, text="Embeddings Data", padding="5")
        self.embeddings_text_area = scrolledtext.ScrolledText(
            embeddings_display_lf, wrap=tk.NONE, width=40, height=SCROLLED_TEXT_HEIGHT,
            bg=COLOR_SURFACE, fg=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY[0], FONT_SIZE_NORMAL),
            relief="flat", borderwidth=0,
            padx=5, pady=5
        )
        self.embeddings_text_area.pack(fill=tk.BOTH, expand=True)
        paned_window.add(embeddings_display_lf, weight=1)

        # Labels Display Area
        labels_display_lf = ttk.LabelFrame(paned_window, text="Labels Data", padding="5")
        self.labels_text_area = scrolledtext.ScrolledText(
            labels_display_lf, wrap=tk.NONE, width=40, height=SCROLLED_TEXT_HEIGHT,
            bg=COLOR_SURFACE, fg=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY[0], FONT_SIZE_NORMAL),
            relief="flat", borderwidth=0,
            padx=5, pady=5
        )
        self.labels_text_area.pack(fill=tk.BOTH, expand=True)
        paned_window.add(labels_display_lf, weight=1)

    def _load_and_display_pickle(self, file_path, text_area_widget, file_label_widget, file_type_hint):
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                text_area_widget.delete(1.0, tk.END)
                text_area_widget.insert(tk.END, f"File: {os.path.basename(file_path)}\n")
                text_area_widget.insert(tk.END, f"Full Path: {file_path}\n\n")
                current_context = self.current_label_context_type if file_type_hint == "Labels" else None
                text_area_widget.insert(tk.END, self.format_data_for_display(data, file_type_hint=file_type_hint, context_type=current_context))
                file_label_widget.config(text=f"Selected: {os.path.basename(file_path)}")
                return data
            except Exception as e:
                text_area_widget.delete(1.0, tk.END)
                text_area_widget.insert(tk.END, f"Error loading/displaying {os.path.basename(file_path)}:\n{str(e)}")
                file_label_widget.config(text=f"Selected: Error")
        return None

    def open_embeddings_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Embedding Pickle File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            self.embeddings_file_path = file_path
            self._load_and_display_pickle(
                self.embeddings_file_path, 
                self.embeddings_text_area, 
                self.embeddings_file_label,
                "Embeddings"
            )

    def open_labels_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Label Pickle File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            self.labels_file_path = file_path
            # Infer context for labels
            p = Path(file_path)
            if p.name.lower() == "labels_all_combined.pkl":
                self.current_label_context_type = "all_combined"
                print(f"Label context inferred: all_combined")
            else:
                parent_dir_name = p.parent.name
                self.current_label_context_type = DIR_TO_LOG_TYPE_VIEWER.get(parent_dir_name, parent_dir_name)
                print(f"Label context inferred from dir '{parent_dir_name}': {self.current_label_context_type}")
                if self.current_label_context_type not in LOG_TYPE_ATTACKS_VIEWER:
                    print(f"Warning: Context '{self.current_label_context_type}' not in LOG_TYPE_ATTACKS_VIEWER. Attack names may not display correctly.")
                    # We might still want to proceed and let format_data_for_display handle missing attack list
            
            self._load_and_display_pickle(
                self.labels_file_path,
                self.labels_text_area,
                self.labels_file_label,
                "Labels" # Hint that this is a labels file
            )
    
    def format_data_for_display(self, data, level=0, file_type_hint="Data", context_type=None):
        indent = "  " * level
        result = ""

        if isinstance(data, np.ndarray):            
            original_printoptions = np.get_printoptions()
            try:
                # Set linewidth to a large value to prevent NumPy from auto-wrapping lines
                # Using a large finite number is often safer than np.inf across versions/environments
                np.set_printoptions(linewidth=10000) 

                result += f"{indent}{file_type_hint} NumPy array:\n"
                result += f"{indent}  Shape: {data.shape}\n"
                result += f"{indent}  Data type: {data.dtype}\n"
                
                attack_names_list = []
                if file_type_hint == "Labels" and context_type:
                    if context_type == "all_combined":
                        attack_names_list = ALL_UNIQUE_ATTACKS_VIEWER
                    elif context_type in LOG_TYPE_ATTACKS_VIEWER:
                        attack_names_list = LOG_TYPE_ATTACKS_VIEWER[context_type]
                    else:
                        # This case means context_type was a directory name not mapping to a known logical type with attacks
                        result += f"{indent}  Warning: Unknown label context '{context_type}'. Cannot display attack names.\n"

                if data.size < NUMPY_DISPLAY_FULL_THRESHOLD_ITEMS and data.size > 0:
                    result += f"{indent}  Values (full array):\n"
                    array_str_lines = str(data).split('\n')
                    for i, line_str in enumerate(array_str_lines):
                        line_display = f"{indent}    {line_str.strip()}"
                        if attack_names_list and i < data.shape[0]: # Ensure we are within bounds for multi-line array prints
                            row_vector = data[i] if data.ndim > 1 else data # Handle 1D and 2D cases for fetching vector
                            active_attacks = [attack_names_list[j] for j, val in enumerate(row_vector) if val == 1 and j < len(attack_names_list)]
                            line_display += f" -> Attacks: {(', '.join(active_attacks) if active_attacks else 'None')}\n"
                        else:
                            line_display += "\n"
                        result += line_display
                elif data.ndim == 0: # Scalar
                    result += f"{indent}  Value: {data.item()}\n"
                elif data.ndim == 1: # Single vector (potentially a single label entry)
                    sample_str = str(data[:NUMPY_SAMPLE_ROWS])
                    result += f"{indent}  Sample ({len(data[:NUMPY_SAMPLE_ROWS])} of {data.shape[0]} items): {sample_str}"
                    if attack_names_list:
                        active_attacks = [attack_names_list[j] for j, val in enumerate(data) if val == 1 and j < len(attack_names_list)]
                        result += f" -> Attacks: {(', '.join(active_attacks) if active_attacks else 'None')}"
                    result += "\n"
                    if len(data) > NUMPY_SAMPLE_ROWS:
                        result += f"{indent}  ... (and more items)\n"
                elif data.ndim >= 2:
                    result += f"{indent}  Sample ({min(NUMPY_SAMPLE_ROWS, data.shape[0])} of {data.shape[0]} rows):\n"
                    for i in range(min(NUMPY_SAMPLE_ROWS, data.shape[0])):
                        row_vector = data[i, :NUMPY_SAMPLE_COLS]
                        row_str = str(row_vector) 
                        if data.shape[1] > NUMPY_SAMPLE_COLS:
                            row_str += " ..."
                        
                        line_display = f"{indent}    Row {i}: {row_str}"
                        if attack_names_list:
                            # For sampled rows, use the full row from original data to get all attack flags
                            full_row_vector_for_attacks = data[i]
                            active_attacks = [attack_names_list[j] for j, val in enumerate(full_row_vector_for_attacks) if val == 1 and j < len(attack_names_list)]
                            line_display += f" -> Attacks: {(', '.join(active_attacks) if active_attacks else 'None')}"
                        line_display += "\n"
                        result += line_display
                        
                    if data.shape[0] > NUMPY_SAMPLE_ROWS:
                        result += f"{indent}  ... (and more rows)\n"
            finally:
                np.set_printoptions(**original_printoptions) # Restore original options
                
        elif isinstance(data, dict):
            result += f"{indent}{file_type_hint} Dictionary with {len(data)} items:\n"
            count = 0
            for key, value in data.items():
                if count < COLLECTION_SAMPLE_ITEMS or (level > 0 and count < 3): # Show more for top-level dicts
                    result += f"{indent}  Key: {str(key)}\n"
                    result += self.format_data_for_display(value, level + 1, file_type_hint="Value")
                else:
                    result += f"{indent}  ... (and more items)\n"
                    break
                count +=1
        elif isinstance(data, list):
            result += f"{indent}{file_type_hint} List with {len(data)} items:\n"
            for i, item in enumerate(data[:COLLECTION_SAMPLE_ITEMS]):
                result += f"{indent}  [{i}]:\n"
                result += self.format_data_for_display(item, level + 1, file_type_hint="Item")
            if len(data) > COLLECTION_SAMPLE_ITEMS:
                result += f"{indent}  ... (and {len(data) - COLLECTION_SAMPLE_ITEMS} more items)\n"
        elif isinstance(data, tuple):
            result += f"{indent}{file_type_hint} Tuple with {len(data)} items:\n"
            for i, item in enumerate(data[:COLLECTION_SAMPLE_ITEMS]):
                result += f"{indent}  [{i}]:\n"
                result += self.format_data_for_display(item, level + 1, file_type_hint="Item")
            if len(data) > COLLECTION_SAMPLE_ITEMS:
                result += f"{indent}  ... (and {len(data) - COLLECTION_SAMPLE_ITEMS} more items)\n"
        else:
            # Truncate long strings
            str_data = str(data)
            if len(str_data) > STRING_TRUNCATE_LIMIT:
                str_data = str_data[:STRING_TRUNCATE_LIMIT] + " ... (truncated)"
            result += f"{indent}{file_type_hint} Value: {str_data}\n"
        
        return result

if __name__ == "__main__":
    root = tk.Tk()
    app = PickleViewer(root)
    root.mainloop()