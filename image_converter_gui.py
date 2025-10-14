#!/usr/bin/env python3
"""
GUI Image-to-Image Converter using Google Gemini API
A user-friendly interface with text box for prompts and file pickers.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image, ImageTk
from io import BytesIO
from dotenv import load_dotenv
import threading

# Load environment variables from .env file
load_dotenv()

class ImageConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image-to-Image Converter - Gemini API")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variables
        self.input_image_path = tk.StringVar()
        self.output_image_path = tk.StringVar()
        self.prompt_text = tk.StringVar()
        self.is_processing = False
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize Gemini client
        self.client = self.setup_client()
        
    def setup_gui(self):
        """Create the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Image-to-Image Converter", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input image selection
        ttk.Label(main_frame, text="Input Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_image_path, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input_image).grid(
            row=1, column=2, pady=5)
        
        # Output image selection
        ttk.Label(main_frame, text="Output Image:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_image_path, width=50).grid(
            row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_image).grid(
            row=2, column=2, pady=5)
        
        # Prompt section
        ttk.Label(main_frame, text="Transformation Prompt:", 
                 font=("Arial", 12, "bold")).grid(row=3, column=0, sticky=tk.W, pady=(20, 5))
        
        # Large text area for prompt
        self.prompt_text_widget = scrolledtext.ScrolledText(
            main_frame, height=8, width=70, wrap=tk.WORD)
        self.prompt_text_widget.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Set default prompt
        default_prompt = """Transform this image with a creative artistic style. 
You can be specific about:
- Art style (e.g., "watercolor painting", "oil painting", "digital art")
- Color scheme (e.g., "black and white", "vibrant colors", "pastel tones")
- Mood (e.g., "dramatic lighting", "soft and dreamy", "cyberpunk aesthetic")
- Objects to add/remove (e.g., "add a sunset", "make it look vintage")
- Scene changes (e.g., "transform into a fantasy forest", "make it look like a movie poster")"""
        
        self.prompt_text_widget.insert(tk.END, default_prompt)
        
        # Example prompts
        examples_frame = ttk.LabelFrame(main_frame, text="Example Prompts (click to use)", padding="5")
        examples_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        example_prompts = [
            "Transform this into a watercolor painting",
            "Make this look like a vintage photograph with sepia tones",
            "Apply a cyberpunk aesthetic with neon colors",
            "Convert to black and white with dramatic lighting",
            "Transform into a fantasy forest scene",
            "Make this look like a movie poster"
        ]
        
        for i, prompt in enumerate(example_prompts):
            btn = ttk.Button(examples_frame, text=prompt, 
                           command=lambda p=prompt: self.set_prompt(p))
            btn.grid(row=i//2, column=i%2, sticky=(tk.W, tk.E), padx=2, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=20)
        
        self.convert_button = ttk.Button(button_frame, text="Convert Image", 
                                       command=self.convert_image, style="Accent.TButton")
        self.convert_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to convert images")
        self.status_label.grid(row=8, column=0, columnspan=3, pady=5)
        
    def setup_client(self):
        """Initialize the Gemini client"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                messagebox.showerror("API Key Error", 
                    "GOOGLE_API_KEY not found in environment variables.\n"
                    "Please add your API key to the .env file:\n"
                    "GOOGLE_API_KEY=your-api-key-here")
                return None
            
            client = genai.Client(api_key=api_key)
            return client
        except Exception as e:
            messagebox.showerror("Client Error", f"Error initializing Gemini client: {e}")
            return None
    
    def browse_input_image(self):
        """Browse for input image file"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=filetypes
        )
        if filename:
            self.input_image_path.set(filename)
            # Auto-set output path
            if not self.output_image_path.get():
                base_name = Path(filename).stem
                output_path = f"{base_name}_converted.png"
                self.output_image_path.set(output_path)
    
    def browse_output_image(self):
        """Browse for output image file"""
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        filename = filedialog.asksaveasfilename(
            title="Save Output Image As",
            filetypes=filetypes,
            defaultextension=".png"
        )
        if filename:
            self.output_image_path.set(filename)
    
    def set_prompt(self, prompt):
        """Set the prompt text from example buttons"""
        self.prompt_text_widget.delete(1.0, tk.END)
        self.prompt_text_widget.insert(tk.END, prompt)
    
    def get_prompt(self):
        """Get the current prompt text"""
        return self.prompt_text_widget.get(1.0, tk.END).strip()
    
    def convert_image(self):
        """Convert the image using the prompt"""
        if self.is_processing:
            return
        
        # Validate inputs
        if not self.input_image_path.get():
            messagebox.showerror("Error", "Please select an input image")
            return
        
        if not os.path.exists(self.input_image_path.get()):
            messagebox.showerror("Error", "Input image file not found")
            return
        
        if not self.output_image_path.get():
            messagebox.showerror("Error", "Please specify an output image path")
            return
        
        prompt = self.get_prompt()
        if not prompt:
            messagebox.showerror("Error", "Please enter a transformation prompt")
            return
        
        if not self.client:
            messagebox.showerror("Error", "Gemini client not initialized")
            return
        
        # Start conversion in a separate thread
        self.is_processing = True
        self.convert_button.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="Converting image...")
        
        thread = threading.Thread(target=self._convert_image_thread)
        thread.daemon = True
        thread.start()
    
    def _convert_image_thread(self):
        """Convert image in a separate thread"""
        try:
            input_path = self.input_image_path.get()
            output_path = self.output_image_path.get()
            prompt = self.get_prompt()
            
            # Load the input image
            image = Image.open(input_path)
            
            # Generate content with both prompt and image
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[prompt, image],
            )
            
            # Process the response
            success = False
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(f"Generated text: {part.text}")
                elif part.inline_data is not None:
                    # Save the generated image
                    generated_image = Image.open(BytesIO(part.inline_data.data))
                    generated_image.save(output_path)
                    success = True
                    break
            
            # Update UI in main thread
            self.root.after(0, self._conversion_complete, success, output_path)
            
        except Exception as e:
            error_msg = f"Error during conversion: {e}"
            self.root.after(0, self._conversion_error, error_msg)
    
    def _conversion_complete(self, success, output_path):
        """Handle conversion completion"""
        self.is_processing = False
        self.convert_button.config(state='normal')
        self.progress.stop()
        
        if success:
            self.status_label.config(text=f"✅ Conversion completed! Saved to: {output_path}")
            messagebox.showinfo("Success", f"Image converted successfully!\nSaved to: {output_path}")
        else:
            self.status_label.config(text="❌ No image was generated")
            messagebox.showerror("Error", "No image was generated in the response")
    
    def _conversion_error(self, error_msg):
        """Handle conversion error"""
        self.is_processing = False
        self.convert_button.config(state='normal')
        self.progress.stop()
        self.status_label.config(text="❌ Conversion failed")
        messagebox.showerror("Error", error_msg)
    
    def clear_all(self):
        """Clear all inputs"""
        self.input_image_path.set("")
        self.output_image_path.set("")
        self.prompt_text_widget.delete(1.0, tk.END)
        self.status_label.config(text="Ready to convert images")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    
    # Set a modern style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Create the application
    app = ImageConverterGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
