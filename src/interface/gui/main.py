"""
GUI interface for PDF Question Answering System.
"""

import sys
from pathlib import Path
from .pdf_qa import PDFQA
import tkinter as tk

def main():
    # Create the main window
    root = tk.Tk()
    root.title("PDF Question Answering System")
    
    # Set window size and position
    window_width = 800
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # Initialize the PDF QA interface
    app = PDFQA(root)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main() 