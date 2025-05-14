#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive TFRecord Viewer GUI with Tree Navigation and Manual Chunked Display
----------------------------------------------------------------------------------
Provides a Tkinter-based GUI to browse TFRecord files.
Displays records in chunks via a "Load More" button. 
Log lines and their corresponding JSON labels are displayed directly and cleanly,
with no wrapping for long JSON labels, and no per-record headers in the text area.
Styling inspired by contemporary UI trends.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import tensorflow as tf
import json
from pathlib import Path
import os

# --- Constants and Style Definitions ---
SCROLLED_TEXT_HEIGHT = 28 
RECORDS_CHUNK_SIZE = 100 
TREEVIEW_WIDTH = 320
# LINE_NUMBER_WIDTH = 4 # Removed

COLOR_BACKGROUND = "#F7F9FC"
COLOR_TEXT_PRIMARY = "#202124"
COLOR_TEXT_SECONDARY = "#5F6368"
COLOR_ACCENT = "#1A73E8"
COLOR_ACCENT_HOVER = "#185ABC"
COLOR_SURFACE = "#FFFFFF"
COLOR_BORDER = "#DADCE0"
COLOR_TREE_SELECTED_BG = "#E8F0FE"
COLOR_STATUS_BAR_BG = "#F1F3F4"
FONT_FAMILY = ("Roboto", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif")
FONT_SIZE_NORMAL = 10
FONT_SIZE_LARGE = 13
FONT_SIZE_BUTTON = 10

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "processed"

class TFRecordViewer:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title("TFRecord Viewer - Clean Display") # Updated title
        self.root.geometry("1450x950") 
        self.root.configure(bg=COLOR_BACKGROUND)

        self.item_to_path = {}
        self.current_file_path = None
        self.current_tf_dataset_for_iteration = None
        self.current_dataset_iterator = None
        self.total_records_in_current_file = -1 # -1 indicates unknown total
        self.loaded_records_count = 0
        self.is_loading_chunk = False

        # self.log_line_numbers = None # Removed
        # self.label_line_numbers = None # Removed

        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure('.', background=COLOR_BACKGROUND, foreground=COLOR_TEXT_PRIMARY, font=(FONT_FAMILY[0], FONT_SIZE_NORMAL))
        self.style.configure("TFrame", background=COLOR_BACKGROUND)
        
        self.style.configure("TLabel", background=COLOR_BACKGROUND, foreground=COLOR_TEXT_PRIMARY, font=(FONT_FAMILY[0], FONT_SIZE_NORMAL))
        self.style.configure("Header.TLabel", font=(FONT_FAMILY[0], FONT_SIZE_LARGE, "bold")) # For potential future use
        self.style.configure("Status.TLabel", background=COLOR_STATUS_BAR_BG, foreground=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY[0], FONT_SIZE_NORMAL -1))
        
        self.style.configure("TButton", 
                        background=COLOR_ACCENT, 
                        foreground=COLOR_SURFACE, 
                        font=(FONT_FAMILY[0], FONT_SIZE_BUTTON, "bold"),
                        padding=(12, 6), # Adjusted padding
                        relief=tk.FLAT,
                        borderwidth=0)
        self.style.map("TButton",
                  background=[('active', COLOR_ACCENT_HOVER), ('pressed', COLOR_ACCENT_HOVER)],
                  relief=[('pressed', tk.FLAT), ('!pressed', tk.FLAT)]) # Keep flat

        self.style.configure("TLabelFrame", 
                        background=COLOR_SURFACE, 
                        foreground=COLOR_TEXT_PRIMARY, 
                        labelmargins=(10, 5),
                        relief=tk.SOLID, # Changed from groove
                        borderwidth=1,
                        bordercolor=COLOR_BORDER, # Explicit border color
                        font=(FONT_FAMILY[0], FONT_SIZE_NORMAL, "bold"))
        # For the text label of the LabelFrame itself:
        self.style.configure("TLabelFrame.Label", 
                        background=COLOR_SURFACE, # Match LabelFrame background
                        foreground=COLOR_TEXT_PRIMARY,
                        font=(FONT_FAMILY[0], FONT_SIZE_NORMAL, "bold"))

        self.style.configure("TPanedwindow", background=COLOR_BACKGROUND)
        self.style.configure("Treeview", 
                        rowheight=24, # Increased row height
                        font=(FONT_FAMILY[0], FONT_SIZE_NORMAL),
                        background=COLOR_SURFACE,
                        fieldbackground=COLOR_SURFACE,
                        foreground=COLOR_TEXT_SECONDARY)
        self.style.map("Treeview",
                  background=[('selected', COLOR_TREE_SELECTED_BG)],
                  foreground=[('selected', COLOR_ACCENT)]) # Selected text to accent color
        
        self.style.configure("Vertical.TScrollbar", background=COLOR_SURFACE, troughcolor=COLOR_BACKGROUND, bordercolor=COLOR_BORDER, arrowcolor=COLOR_TEXT_SECONDARY)
        self.style.configure("Horizontal.TScrollbar", background=COLOR_SURFACE, troughcolor=COLOR_BACKGROUND, bordercolor=COLOR_BORDER, arrowcolor=COLOR_TEXT_SECONDARY)

        try:
            # Try to make sashes thin and less obtrusive
            self.style.configure("Sash", background=COLOR_BORDER, gripcount=0, sashthickness=3, borderwidth=0, relief=tk.FLAT)
            self.style.configure("Vertical.Sash", background=COLOR_BORDER, gripcount=0, sashthickness=3, borderwidth=0, relief=tk.FLAT)
            self.style.configure("Horizontal.Sash", background=COLOR_BORDER, gripcount=0, sashthickness=3, borderwidth=0, relief=tk.FLAT)
        except tk.TclError:
            print("Warning: Advanced Sash styling not fully supported by current theme/platform.")

        self.create_widgets()
        self.populate_treeview(PROCESSED_DIR, "")

    def create_widgets(self):
        main_paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=15, pady=(15,5))

        tree_lf = ttk.LabelFrame(main_paned_window, text="TFRecord Explorer", padding=(10,10))
        self.tree = ttk.Treeview(tree_lf, columns=("fullpath",), displaycolumns=(), selectmode="browse", style="Treeview")
        self.tree.heading("#0", text="Project Files", anchor='w')
        ysb_tree = ttk.Scrollbar(tree_lf, orient='vertical', command=self.tree.yview, style="Vertical.TScrollbar")
        xsb_tree = ttk.Scrollbar(tree_lf, orient='horizontal', command=self.tree.xview, style="Horizontal.TScrollbar")
        self.tree.configure(yscrollcommand=ysb_tree.set, xscrollcommand=xsb_tree.set)
        self.tree.grid(row=0, column=0, sticky='nsew'); ysb_tree.grid(row=0, column=1, sticky='ns'); xsb_tree.grid(row=1, column=0, sticky='ew')
        tree_lf.grid_rowconfigure(0, weight=1); tree_lf.grid_columnconfigure(0, weight=1)
        main_paned_window.add(tree_lf, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        data_display_frame = ttk.Frame(main_paned_window)
        right_content_frame = ttk.Frame(data_display_frame)
        right_content_frame.pack(fill=tk.BOTH, expand=True, padx=(10,0))

        data_paned_window = ttk.PanedWindow(right_content_frame, orient=tk.HORIZONTAL)
        data_paned_window.pack(fill=tk.BOTH, expand=True, pady=(0,5))

        logs_display_lf = ttk.LabelFrame(data_paned_window, text="Log Record Content", padding="8")
        self.logs_text_area = scrolledtext.ScrolledText(
            logs_display_lf, wrap=tk.NONE, height=SCROLLED_TEXT_HEIGHT,
            bg=COLOR_SURFACE, fg=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY[0], FONT_SIZE_NORMAL),
            relief=tk.FLAT, borderwidth=1, highlightthickness=1, highlightbackground=COLOR_BORDER, highlightcolor=COLOR_ACCENT,
            padx=10, pady=10)
        self.logs_text_area.pack(fill=tk.BOTH, expand=True)
        data_paned_window.add(logs_display_lf, weight=3)

        labels_display_lf = ttk.LabelFrame(data_paned_window, text="Associated Labels (JSON)", padding="8")
        self.labels_text_area = scrolledtext.ScrolledText(
            labels_display_lf, wrap=tk.NONE, height=SCROLLED_TEXT_HEIGHT,
            bg=COLOR_SURFACE, fg=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY[0], FONT_SIZE_NORMAL),
            relief=tk.FLAT, borderwidth=1, highlightthickness=1, highlightbackground=COLOR_BORDER, highlightcolor=COLOR_ACCENT,
            padx=10, pady=10)
        self.labels_text_area.pack(fill=tk.BOTH, expand=True)
        data_paned_window.add(labels_display_lf, weight=2)
        
        self.load_more_button = ttk.Button(right_content_frame, text="Load More Records (%s)" % RECORDS_CHUNK_SIZE, 
                                           command=self.load_next_chunk_of_records, state=tk.DISABLED)
        self.load_more_button.pack(side=tk.BOTTOM, pady=(10,5))

        main_paned_window.add(data_display_frame, weight=3)
        main_paned_window.sashpos(0, TREEVIEW_WIDTH)

        status_bar_frame = ttk.Frame(self.root, style="TFrame", height=28)
        status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0))
        top_border = ttk.Frame(status_bar_frame, height=1, style="Thin.TFrame")
        self.style.configure("Thin.TFrame", background=COLOR_BORDER)
        top_border.pack(side=tk.TOP, fill=tk.X)
        self.status_bar_label = ttk.Label(status_bar_frame, text="Ready", style="Status.TLabel", padding=(10,3))
        self.status_bar_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def populate_treeview(self, current_path: Path, parent_item_id):
        if not current_path.exists() or not current_path.is_dir(): return
        for p_item in sorted(current_path.iterdir()):
            if p_item.is_dir():
                dir_id = self.tree.insert(parent_item_id, 'end', text=p_item.name, open=False)
                self.item_to_path[dir_id] = str(p_item)
                self.populate_treeview(p_item, dir_id)
            elif p_item.is_file() and p_item.name.endswith(".tfrecord"):
                file_id = self.tree.insert(parent_item_id, 'end', text=p_item.name, values=(str(p_item),))
                self.item_to_path[file_id] = str(p_item)

    def on_tree_select(self, event):
        if self.is_loading_chunk: return
        selected_item_id = self.tree.selection()
        if not selected_item_id: return
        item_id = selected_item_id[0]
        file_path_str = self.item_to_path.get(item_id)

        if file_path_str and Path(file_path_str).is_file() and file_path_str.endswith(".tfrecord"):
            self.current_file_path = Path(file_path_str)
            # Display file info in status bar, not in text areas
            self.status_bar_label.config(text=f"File: {self.current_file_path.name} | Path: {self.current_file_path}")
            self.root.update_idletasks()
            self.logs_text_area.config(state=tk.NORMAL); self.labels_text_area.config(state=tk.NORMAL)
            self.logs_text_area.delete(1.0, tk.END); self.labels_text_area.delete(1.0, tk.END)
            self.loaded_records_count = 0
            self.total_records_in_current_file = -1 
            self.current_tf_dataset_for_iteration = None 
            if self.current_dataset_iterator: self.current_dataset_iterator = None 
            
            self.is_loading_chunk = True
            try:
                self.current_tf_dataset_for_iteration = tf.data.TFRecordDataset([str(self.current_file_path)], compression_type='GZIP')
                # Defer total record count, load first chunk immediately
                self.status_bar_label.config(text=f"Selected: {self.current_file_path.name}. Loading initial {RECORDS_CHUNK_SIZE} records...")
                self.load_next_chunk_of_records() # Load initial chunk
            except Exception as e:
                self.status_bar_label.config(text=f"Error initializing file: {e}")
                if self.logs_text_area: self.logs_text_area.insert(tk.END, f"Error: {e}")
                self.load_more_button.config(state=tk.DISABLED)
            finally:
                self.is_loading_chunk = False 
                if self.logs_text_area: self.logs_text_area.config(state=tk.DISABLED)
                if self.labels_text_area: self.labels_text_area.config(state=tk.DISABLED)
        else:
            self.status_bar_label.config(text=f"Selected directory: {Path(file_path_str).name if file_path_str else 'None'}")
            self.load_more_button.config(state=tk.DISABLED)
            self.current_file_path = None

    def load_next_chunk_of_records(self):
        if self.is_loading_chunk and self.loaded_records_count > 0: return 
        
        if not self.current_tf_dataset_for_iteration:
             self.load_more_button.config(state=tk.DISABLED)
             return # No dataset selected or available

        if not self.current_dataset_iterator: # First load for this file, or re-opening
            try:
                self.current_dataset_iterator = iter(self.current_tf_dataset_for_iteration)
            except Exception as e:
                self.status_bar_label.config(text=f"Error creating iterator: {e}")
                self.is_loading_chunk = False
                self.load_more_button.config(state=tk.DISABLED)
                return

        if self.total_records_in_current_file != -1 and self.loaded_records_count >= self.total_records_in_current_file:
            self.load_more_button.config(state=tk.DISABLED)
            self.status_bar_label.config(text=f"All {self.total_records_in_current_file} records displayed for {self.current_file_path.name}.")
            return

        self.is_loading_chunk = True
        self.load_more_button.config(state=tk.DISABLED)
        
        current_total_str = str(self.total_records_in_current_file) if self.total_records_in_current_file != -1 else "unknown"
        self.status_bar_label.config(text=f"Loading records {self.loaded_records_count + 1}-{min(self.loaded_records_count + RECORDS_CHUNK_SIZE, (self.total_records_in_current_file if self.total_records_in_current_file != -1 else float('inf' )))} of {current_total_str}...")
        self.root.update_idletasks()
        
        records_to_fetch_this_chunk = []
        initial_loaded_count_for_this_call = self.loaded_records_count
        eof_reached_this_chunk = False
        try:
            for _ in range(RECORDS_CHUNK_SIZE):
                raw_record = next(self.current_dataset_iterator)
                feature_description = {'l': tf.io.FixedLenFeature([], tf.string), 'y': tf.io.FixedLenFeature([], tf.string)}
                example = tf.io.parse_single_example(raw_record, feature_description)
                log_line = example['l'].numpy().decode('utf-8', errors='replace')
                labels_json_str = example['y'].numpy().decode('utf-8', errors='replace')
                try: 
                    labels_data = json.loads(labels_json_str)
                    # Ensure labels_str is a single line, no indent for pretty print
                    formatted_labels_str = json.dumps(labels_data) # NO INDENT
                except json.JSONDecodeError: 
                    formatted_labels_str = labels_json_str 
                records_to_fetch_this_chunk.append((log_line, formatted_labels_str))
                self.loaded_records_count += 1
        except StopIteration: 
            eof_reached_this_chunk = True
            if self.total_records_in_current_file == -1: 
                self.total_records_in_current_file = self.loaded_records_count
            # Status update will happen below based on eof_reached_this_chunk
        except Exception as e: 
            self.status_bar_label.config(text=f"Error loading chunk: {e}")
            if self.logs_text_area: self.logs_text_area.config(state=tk.NORMAL); self.logs_text_area.insert(tk.END, f"\nError during chunk loading: {e}"); self.logs_text_area.config(state=tk.DISABLED)
        
        if records_to_fetch_this_chunk:
            self.display_records(records_to_fetch_this_chunk, append=(initial_loaded_count_for_this_call > 0))

        if not eof_reached_this_chunk and (self.total_records_in_current_file == -1 or self.loaded_records_count < self.total_records_in_current_file):
            self.load_more_button.config(state=tk.NORMAL)
            current_total_str = str(self.total_records_in_current_file) if self.total_records_in_current_file != -1 else "many"
            self.status_bar_label.config(text=f"Displayed {self.loaded_records_count} of {current_total_str}. Click 'Load More'.")
        else: 
            self.load_more_button.config(state=tk.DISABLED)
            if self.current_file_path:
                 self.status_bar_label.config(text=f"All {self.loaded_records_count} records displayed for {self.current_file_path.name}.")
        
        self.is_loading_chunk = False

    def display_records(self, records_chunk, append=False):
        if self.logs_text_area: self.logs_text_area.config(state=tk.NORMAL)
        if self.labels_text_area: self.labels_text_area.config(state=tk.NORMAL)
        
        if not append: 
            if self.logs_text_area: self.logs_text_area.delete(1.0, tk.END)
            if self.labels_text_area: self.labels_text_area.delete(1.0, tk.END)

        if not records_chunk and not append: 
            if self.logs_text_area: self.logs_text_area.insert(tk.END, "No records to display for this chunk.")
        
        for i, (log_line, labels_str) in enumerate(records_chunk):
            # Display content directly without record headers
            if self.logs_text_area: self.logs_text_area.insert(tk.END, log_line + "\n")
            if self.labels_text_area: self.labels_text_area.insert(tk.END, labels_str + "\n") # Ensure newline after each label string
            
        if append: 
            if self.logs_text_area: self.logs_text_area.see(tk.END)
            if self.labels_text_area: self.labels_text_area.see(tk.END)
        
        if self.logs_text_area: self.logs_text_area.config(state=tk.DISABLED)
        if self.labels_text_area: self.labels_text_area.config(state=tk.DISABLED)

def main_gui():
    root = tk.Tk()
    app = TFRecordViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main_gui()