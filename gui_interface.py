import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
from datetime import datetime
import threading
import asyncio
import edge_tts
import os
import tempfile
import pygame
import speech_recognition as sr

class RAGTeachingAssistant:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG AI Teaching Assistant")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e1e")
        
        # Make window fullscreen
        self.root.state('zoomed')  # Windows ke liye
        # self.root.attributes('-zoomed', True)  # Linux ke liye uncomment kar dena
        
        # Bring window to front and focus
        self.root.lift()
        self.root.focus_force()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))
        
        # Load embeddings
        try:
            self.df = joblib.load('embedding.joblib')
            self.embeddings_loaded = True
        except Exception as e:
            self.embeddings_loaded = False
            messagebox.showerror("Error", f"Could not load embeddings: {str(e)}")
        
        # Initialize voice components
        pygame.mixer.init()
        self.recognizer = sr.Recognizer()
        self.is_speaking = False
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Title Section
        title_frame = tk.Frame(self.root, bg="#2d2d2d", height=70)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🎓 RAG AI Teaching Assistant",
            font=("Helvetica", 20, "bold"),
            bg="#2d2d2d",
            fg="#4CAF50"
        )
        title_label.pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg="#1e1e1e")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Chat History Section
        history_label = tk.Label(
            main_container,
            text="Chat History",
            font=("Helvetica", 12, "bold"),
            bg="#1e1e1e",
            fg="#ffffff"
        )
        history_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Chat display with custom styling
        self.chat_display = scrolledtext.ScrolledText(
            main_container,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#2d2d2d",
            fg="#e0e0e0",
            insertbackground="#4CAF50",
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_display.config(state=tk.DISABLED)
        
        # Configure tags for styling
        self.chat_display.tag_config("user", foreground="#4CAF50", font=("Helvetica", 10, "bold"))
        self.chat_display.tag_config("assistant", foreground="#2196F3", font=("Helvetica", 10, "bold"))
        self.chat_display.tag_config("timestamp", foreground="#888888", font=("Helvetica", 8))
        self.chat_display.tag_config("response_text", foreground="#e0e0e0", font=("Consolas", 10))
        
        # Query Input Section
        input_frame = tk.Frame(main_container, bg="#1e1e1e")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        query_label = tk.Label(
            input_frame,
            text="Ask Your Question:",
            font=("Helvetica", 11, "bold"),
            bg="#1e1e1e",
            fg="#ffffff"
        )
        query_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Query entry with frame for border effect
        entry_frame = tk.Frame(input_frame, bg="#4CAF50", bd=0)
        entry_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.query_entry = tk.Text(
            entry_frame,
            height=3,
            font=("Helvetica", 11),
            bg="#2d2d2d",
            fg="#ffffff",
            insertbackground="#4CAF50",
            relief=tk.FLAT,
            padx=10,
            pady=10,
            wrap=tk.WORD
        )
        self.query_entry.pack(fill=tk.BOTH, padx=2, pady=2)
        
        # Bind Enter key (Shift+Enter for new line)
        self.query_entry.bind("<Return>", self.handle_enter_key)
        
        # Buttons Frame
        button_frame = tk.Frame(input_frame, bg="#1e1e1e")
        button_frame.pack(fill=tk.X)
        
        # Ask Button
        self.ask_button = tk.Button(
            button_frame,
            text="Ask Question 🚀",
            command=self.process_query,
            font=("Helvetica", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.ask_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Voice Input Button
        self.voice_button = tk.Button(
            button_frame,
            text="🎤 Voice Input",
            command=self.start_voice_input,
            font=("Helvetica", 11, "bold"),
            bg="#2196F3",
            fg="white",
            activebackground="#1976D2",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.voice_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear Button
        clear_button = tk.Button(
            button_frame,
            text="Clear History 🗑️",
            command=self.clear_chat,
            font=("Helvetica", 11, "bold"),
            bg="#f44336",
            fg="white",
            activebackground="#da190b",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        clear_button.pack(side=tk.LEFT)
        
        # Status Label
        self.status_label = tk.Label(
            button_frame,
            text="Ready" if self.embeddings_loaded else "⚠️ Embeddings not loaded!",
            font=("Helvetica", 9),
            bg="#1e1e1e",
            fg="#4CAF50" if self.embeddings_loaded else "#f44336"
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
    def handle_enter_key(self, event):
        """Handle Enter key press (Shift+Enter for new line)"""
        if event.state & 0x1:  # Shift key is pressed
            return  # Allow default behavior (new line)
        else:
            self.process_query()
            return "break"  # Prevent default behavior
    
    def create_embedding(self, text_list):
        """Create embeddings using Ollama BGE-M3"""
        try:
            r = requests.post("http://localhost:11434/api/embed", json={
                "model": "bge-m3",
                "input": text_list
            })
            embedding = r.json()['embeddings']
            return embedding
        except Exception as e:
            raise Exception(f"Embedding creation failed: {str(e)}")
    
    def inference(self, prompt):
        """Get response from Ollama LLaMA"""
        try:
            r = requests.post("http://localhost:11434/api/generate", json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            })
            response = r.json()
            return response['response']
        except Exception as e:
            raise Exception(f"Inference failed: {str(e)}")
    
    def add_message_to_chat(self, sender, message, timestamp=None):
        """Add a message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add separator if not first message
        if self.chat_display.get("1.0", tk.END).strip():
            self.chat_display.insert(tk.END, "\n" + "─" * 80 + "\n\n")
        
        # Add sender and timestamp
        self.chat_display.insert(tk.END, f"{sender}: ", sender.lower())
        self.chat_display.insert(tk.END, f"[{timestamp}]\n", "timestamp")
        
        # Add message
        self.chat_display.insert(tk.END, f"{message}\n", "response_text")
        
        # Speak response if it's from assistant
        if sender.lower() == "assistant":
            self.speak_text(message)
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def process_query_thread(self, query):
        """Process query in separate thread"""
        try:
            # Update status
            self.status_label.config(text="🔍 Searching...", fg="#FFC107")
            
            # Create embedding for query
            question_embedding = self.create_embedding([query])[0]
            
            # Find similar chunks
            similarities = cosine_similarity(
                np.vstack(self.df['embedding']), 
                [question_embedding]
            ).flatten()
            
            # Get top 3 results
            max_indx = similarities.argsort()[::-1][0:3]
            new_df = self.df.loc[max_indx]
            
            # Update status
            self.status_label.config(text="🤖 Generating response...", fg="#FFC107")
            
            # Create prompt
            prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title,video number,start time in seconds,end time in seconds , the text at that time:

{new_df[["title","number","start","end","text"]].to_json(orient="records")}
-----------------------------------
"{query}"
User asked this question related to the video chunks,you have to answer in a human way (dont mention the above format,its just for you )where and how much content is  taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course 
'''
            
            # Get response
            response = self.inference(prompt)
            
            # Add response to chat
            self.add_message_to_chat("Assistant", response)
            
            # Update status
            self.status_label.config(text="✅ Ready", fg="#4CAF50")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.add_message_to_chat("System", error_msg)
            self.status_label.config(text="❌ Error occurred", fg="#f44336")
        
        finally:
            # Re-enable buttons
            self.ask_button.config(state=tk.NORMAL, text="Ask Question 🚀")
            self.voice_button.config(state=tk.NORMAL)
    
    def process_query(self):
        """Process user query"""
        if not self.embeddings_loaded:
            messagebox.showerror("Error", "Embeddings not loaded. Please check embedding.joblib file.")
            return
        
        # Get query from entry
        query = self.query_entry.get("1.0", tk.END).strip()
        
        if not query:
            messagebox.showwarning("Warning", "Please enter a question!")
            return
        
        # Add user query to chat
        self.add_message_to_chat("User", query)
        
        # Clear entry
        self.query_entry.delete("1.0", tk.END)
        
        # Disable button while processing
        self.ask_button.config(state=tk.DISABLED, text="Processing... ⏳")
        
        # Process in separate thread to keep GUI responsive
        thread = threading.Thread(target=self.process_query_thread, args=(query,))
        thread.daemon = True
        thread.start()
    
    def clear_chat(self):
        """Clear chat history"""
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear chat history?"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.status_label.config(text="✅ Chat cleared", fg="#4CAF50")
    
    async def text_to_speech_async(self, text):
        """Convert text to speech using Edge TTS"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
            
            # Indian English female voice (natural sound)
            # Options: en-IN-NeerjaNeural (female), en-IN-PrabhatNeural (male)
            communicate = edge_tts.Communicate(text, voice="en-IN-NeerjaNeural")
            await communicate.save(temp_file)
            
            # Play audio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            
            # Cleanup
            pygame.mixer.music.unload()
            os.unlink(temp_file)
            
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_speaking = False
    
    def speak_text(self, text):
        """Speak text in separate thread"""
        if self.is_speaking:
            return
        
        self.is_speaking = True
        
        def run_async():
            asyncio.run(self.text_to_speech_async(text))
        
        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()
    
    def listen_voice_command(self):
        """Listen to voice input and convert to text"""
        try:
            self.status_label.config(text="🎤 Listening...", fg="#FFC107")
            self.voice_button.config(state=tk.DISABLED)
            
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                # Convert to text using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                
                # Insert into query entry
                self.query_entry.delete("1.0", tk.END)
                self.query_entry.insert("1.0", text)
                
                self.status_label.config(text="✅ Voice captured", fg="#4CAF50")
                
                # Auto-submit after 1 second
                self.root.after(1000, self.process_query)
                
        except sr.WaitTimeoutError:
            self.status_label.config(text="⏱️ Timeout - No speech detected", fg="#f44336")
        except sr.UnknownValueError:
            self.status_label.config(text="❌ Could not understand", fg="#f44336")
        except Exception as e:
            self.status_label.config(text=f"❌ Error: {str(e)}", fg="#f44336")
        finally:
            self.voice_button.config(state=tk.NORMAL)
    
    def start_voice_input(self):
        """Start voice input in separate thread"""
        thread = threading.Thread(target=self.listen_voice_command)
        thread.daemon = True
        thread.start()

def main():
    root = tk.Tk()
    app = RAGTeachingAssistant(root)
    root.mainloop()

if __name__ == "__main__":
    main()