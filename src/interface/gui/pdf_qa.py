"""
Simple interface for PDF question answering.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from pathlib import Path
import threading
import atexit
# from dask.distributed import Client, LocalCluster  # Removing Dask usage
from concurrent.futures import ThreadPoolExecutor

from ...processing.document import DocumentProcessor

class PDFQA:
    def __init__(self, root):
        """Initialize the PDF QA interface."""
        self.window = root
        self.window.title("PDF Question Answering")
        
        # Initialize parallel processing with ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=4)
        atexit.register(self.cleanup)
        
        self.pdf_processor = DocumentProcessor()
        self.setup_ui()
        
        # Bind window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def cleanup(self):
        """Cleanup parallel processing resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown()
            
    def on_closing(self):
        """Handle window closing event."""
        self.cleanup()
        self.window.destroy()
        
    def setup_ui(self):
        """Setup the user interface."""
        # File selection
        file_frame = ttk.Frame(self.window)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Label(file_frame, text="PDF File:").pack(side=tk.LEFT)
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)
        ttk.Button(file_frame, text="Load", command=self.load_pdf).pack(side=tk.LEFT, padx=5)
        
        # Question input
        question_frame = ttk.Frame(self.window)
        question_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(question_frame, text="Question:").pack(side=tk.LEFT)
        self.question_entry = ttk.Entry(question_frame, width=70)
        self.question_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(question_frame, text="Ask", command=self.ask_question).pack(side=tk.LEFT)
        
        # Results area
        results_frame = ttk.Frame(self.window)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Context display
        ttk.Label(results_frame, text="Relevant Context:").pack(anchor=tk.W)
        self.context_text = scrolledtext.ScrolledText(results_frame, height=10)
        self.context_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.window, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
    def browse_file(self):
        """Open file dialog to select PDF."""
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)
            
    def load_pdf(self):
        """Load the selected PDF file."""
        file_path = self.file_path.get()
        if not file_path:
            self.status_var.set("Please select a PDF file first")
            return
            
        def load():
            self.status_var.set("Loading PDF...")
            success = self.pdf_processor.load_pdf(file_path)
            if success:
                stats = self.pdf_processor.get_document_stats()
                self.status_var.set(
                    f"PDF loaded successfully. "
                    f"Length: {stats['total_length']}, "
                    f"Chunks: {stats['num_chunks']}"
                )
            else:
                self.status_var.set("Error loading PDF")
                
        # Run in background thread
        threading.Thread(target=load).start()
        
    def load_document(self, file_path):
        """Load a document directly (for testing)."""
        self.file_path.set(file_path)
        success = self.pdf_processor.load_pdf(file_path)
        if not success:
            raise RuntimeError("Failed to load PDF")
            
    def ask_question(self, question=None):
        """Process the question and display results."""
        if question is None:
            question = self.question_entry.get()
        if not question:
            self.status_var.set("Please enter a question")
            return None
            
        self.status_var.set("Processing question...")
        try:
            result = self.pdf_processor.answer_question(question)
            
            # Update UI if running in GUI mode
            if question == self.question_entry.get():
                self.context_text.delete(1.0, tk.END)
                if result:
                    self.context_text.insert(tk.END, result)
                    self.status_var.set("Question processed successfully")
                else:
                    self.status_var.set("No relevant answer found")
                    
            return result
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            self.status_var.set(error_msg)
            return None
        
    def run(self):
        """Start the application."""
        self.window.mainloop()

def main():
    app = PDFQA()
    app.run()

if __name__ == "__main__":
    main() 