"""
UPDATED SPEECH TO SIGN CONVERTER
FIXED: Enhanced speech recognition and text-to-speech with multiple fallback methods
FIXED: Sign image generation and serving
"""

import speech_recognition as sr
import pyttsx3
import threading
import time
import queue
import sys
import os
import tempfile
import pygame
from gtts import gTTS
import platform
import cv2
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SpeechToSign:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        self.current_text = ""
        self.update_callback = None
        self.status_callback = None
        
        # Enhanced speech engine initialization
        self.pyttsx_engine = None
        self.init_speech_engine()
        
        # Initialize sign images
        self.sign_images = {}
        self.init_sign_images()
        
        # Calibrate microphone for ambient noise
        print("🔊 Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Microphone calibrated")
        
    def init_speech_engine(self):
        """Initialize multiple speech engines for fallback"""
        try:
            # Initialize pyttsx3 (offline)
            self.pyttsx_engine = pyttsx3.init()
            self.pyttsx_engine.setProperty('rate', 150)
            self.pyttsx_engine.setProperty('volume', 0.8)
            print("✅ pyttsx3 engine initialized")
        except Exception as e:
            print(f"⚠️ pyttsx3 initialization failed: {e}")
            self.pyttsx_engine = None
            
    def init_sign_images(self):
        """Initialize sign images for all letters and special characters"""
        print("🖼️ Initializing sign images...")
        
        # Define all possible sign classes
        sign_classes = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'SPACE', 'ENTER', 'BACKSPACE'
        ]
        
        for sign_class in sign_classes:
            self.sign_images[sign_class] = self.create_sign_image(sign_class)
            
        print(f"✅ Created {len(self.sign_images)} sign images")
    
    def create_sign_image(self, sign_class):
        """Create a sign image for the given class"""
        # Create a blank image
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
        
        if sign_class == 'SPACE':
            # Create space symbol
            cv2.rectangle(img, (50, 80), (150, 120), (0, 0, 0), 2)
            cv2.putText(img, "SPACE", (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        elif sign_class == 'ENTER':
            # Create enter symbol
            cv2.rectangle(img, (50, 80), (150, 120), (0, 0, 0), 2)
            cv2.putText(img, "ENTER", (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        elif sign_class == 'BACKSPACE':
            # Create backspace symbol
            cv2.rectangle(img, (50, 80), (150, 120), (0, 0, 0), 2)
            cv2.putText(img, "BACK", (65, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(img, "SPACE", (60, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            # Create letter image
            # Add a colored background based on letter
            color_map = {
                'A': (255, 200, 200), 'B': (200, 255, 200), 'C': (200, 200, 255),
                'D': (255, 255, 200), 'E': (255, 200, 255), 'F': (200, 255, 255),
                'G': (255, 225, 200), 'H': (225, 255, 200), 'I': (225, 200, 255),
                'J': (200, 225, 255), 'K': (255, 200, 225), 'L': (200, 255, 225),
                'M': (225, 225, 200), 'N': (225, 200, 225), 'O': (200, 225, 225),
                'P': (240, 240, 200), 'Q': (240, 200, 240), 'R': (200, 240, 240),
                'S': (220, 240, 200), 'T': (220, 200, 240), 'U': (200, 220, 240),
                'V': (240, 220, 200), 'W': (240, 200, 220), 'X': (200, 240, 220),
                'Y': (220, 220, 240), 'Z': (240, 220, 220)
            }
            
            bg_color = color_map.get(sign_class, (240, 240, 240))
            img[:, :] = bg_color
            
            # Add letter in the center
            text_size = cv2.getTextSize(sign_class, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
            text_x = (200 - text_size[0]) // 2
            text_y = (200 + text_size[1]) // 2
            
            # Add shadow
            cv2.putText(img, sign_class, (text_x+3, text_y+3), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 100, 100), 5)
            # Add main text
            cv2.putText(img, sign_class, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
            
            # Add border
            cv2.rectangle(img, (5, 5), (195, 195), (0, 0, 0), 2)
        
        return img
    
    def text_to_gloss_sequence(self, text):
        """Convert text to sign sequence"""
        if not text:
            return []
            
        print(f"🔤 Converting text to sign sequence: '{text}'")
        
        sequence = []
        words = text.upper().split()
        
        for i, word in enumerate(words):
            # Add letters of the word
            for letter in word:
                if letter.isalpha():
                    sequence.append(letter)
            
            # Add space between words (but not after the last word)
            if i < len(words) - 1:
                sequence.append('SPACE')
        
        print(f"✅ Generated sign sequence: {sequence}")
        return sequence
        
    def set_callbacks(self, update_callback, status_callback):
        """Set callbacks for UI updates"""
        self.update_callback = update_callback
        self.status_callback = status_callback
        
    def enhanced_speech_synthesis(self, text, language='en'):
        """Enhanced speech synthesis using multiple methods"""
        try:
            print(f"🔊 Speaking: '{text}' in {language}")
            
            # Method 1: Try pyttsx3 first (works offline)
            if self.pyttsx_engine:
                try:
                    # Set language properties
                    if language == 'hi':
                        # Hindi voice (if available)
                        voices = self.pyttsx_engine.getProperty('voices')
                        for voice in voices:
                            if 'hindi' in voice.name.lower() or 'india' in voice.name.lower():
                                self.pyttsx_engine.setProperty('voice', voice.id)
                                break
                    elif language == 'ta':
                        # Tamil voice (if available)
                        voices = self.pyttsx_engine.getProperty('voices')
                        for voice in voices:
                            if 'tamil' in voice.name.lower() or 'india' in voice.name.lower():
                                self.pyttsx_engine.setProperty('voice', voice.id)
                                break
                    
                    self.pyttsx_engine.say(text)
                    self.pyttsx_engine.runAndWait()
                    print("✅ Speech completed using pyttsx3")
                    return True
                    
                except Exception as e:
                    print(f"⚠️ pyttsx3 speech failed: {e}")
            
            # Method 2: Try gTTS (requires internet)
            try:
                tts = gTTS(text=text, lang=language)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
                    tts.save(tmpfile.name)
                    
                    # Initialize pygame mixer if not already done
                    if not pygame.mixer.get_init():
                        pygame.mixer.init()
                    
                    pygame.mixer.music.load(tmpfile.name)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to complete
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    # Clean up
                    pygame.mixer.music.stop()
                    os.unlink(tmpfile.name)
                
                print("✅ Speech completed using gTTS")
                return True
                
            except Exception as e:
                print(f"⚠️ gTTS speech failed: {e}")
            
            # Method 3: Platform-specific solutions
            system = platform.system()
            if system == "Darwin":  # macOS
                try:
                    os.system(f'say "{text}"')
                    print("✅ Speech completed using macOS say")
                    return True
                except Exception as e:
                    print(f"⚠️ macOS say failed: {e}")
            elif system == "Linux":
                try:
                    os.system(f'espeak "{text}"')
                    print("✅ Speech completed using espeak")
                    return True
                except Exception as e:
                    print(f"⚠️ espeak failed: {e}")
            elif system == "Windows":
                try:
                    import win32com.client
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    speaker.Speak(text)
                    print("✅ Speech completed using SAPI")
                    return True
                except Exception as e:
                    print(f"⚠️ Windows SAPI failed: {e}")
            
            print("❌ All speech synthesis methods failed")
            return False
            
        except Exception as e:
            print(f"❌ Speech synthesis error: {e}")
            return False
    
    def start_listening(self):
        """Start continuous speech recognition"""
        if self.is_listening:
            print("⚠️ Already listening")
            return
            
        self.is_listening = True
        self.current_text = ""
        
        # Start listening in a separate thread
        listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listen_thread.start()
        
        print("🎤 Started listening for speech")
        if self.status_callback:
            self.status_callback({'type': 'listening_started', 'message': 'Started listening'})
    
    def stop_listening(self):
        """Stop speech recognition"""
        self.is_listening = False
        print("🛑 Stopped listening")
        if self.status_callback:
            self.status_callback({'type': 'listening_stopped', 'message': 'Stopped listening'})
    
    def _listen_loop(self):
        """Main listening loop"""
        while self.is_listening:
            try:
                print("👂 Listening...")
                if self.status_callback:
                    self.status_callback({'type': 'listening', 'message': 'Listening...'})
                
                # Listen for audio with timeout
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                print("🔍 Processing speech...")
                if self.status_callback:
                    self.status_callback({'type': 'processing', 'message': 'Processing speech...'})
                
                # Recognize speech using multiple engines
                text = None
                
                # Try Google Speech Recognition first
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"✅ Google Recognition: '{text}'")
                except sr.UnknownValueError:
                    print("❌ Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print(f"❌ Google Speech Recognition error: {e}")
                
                # Fallback to Sphinx (offline)
                if not text:
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                        print(f"✅ Sphinx Recognition: '{text}'")
                    except sr.UnknownValueError:
                        print("❌ Sphinx could not understand audio")
                    except sr.RequestError as e:
                        print(f"❌ Sphinx error: {e}")
                
                if text:
                    self.current_text = text
                    print(f"🎯 Recognized: '{text}'")
                    
                    # Convert to sign sequence
                    sign_sequence = self.text_to_gloss_sequence(text)
                    
                    # Call update callback
                    if self.update_callback:
                        self.update_callback({
                            'text': text,
                            'sign_sequence': sign_sequence,
                            'timestamp': time.time()
                        })
                    
                    if self.status_callback:
                        self.status_callback({
                            'type': 'recognized', 
                            'message': f'Recognized: {text}',
                            'text': text,
                            'sign_sequence': sign_sequence
                        })
                
            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                continue
            except Exception as e:
                print(f"❌ Listening error: {e}")
                if self.status_callback:
                    self.status_callback({'type': 'error', 'message': f'Error: {str(e)}'})
                
                # Brief pause before retrying
                time.sleep(1)
    
    def speak_text(self, text, language='en'):
        """Speak text using available TTS methods"""
        if self.is_speaking:
            print("⚠️ Already speaking")
            return False
            
        self.is_speaking = True
        
        try:
            # Add to speech queue and process in separate thread
            speech_thread = threading.Thread(
                target=self._speak_thread, 
                args=(text, language), 
                daemon=True
            )
            speech_thread.start()
            return True
            
        except Exception as e:
            print(f"❌ Error starting speech: {e}")
            self.is_speaking = False
            return False
    
    def _speak_thread(self, text, language):
        """Thread for speech synthesis"""
        try:
            success = self.enhanced_speech_synthesis(text, language)
            
            if self.status_callback:
                if success:
                    self.status_callback({
                        'type': 'speech_completed', 
                        'message': f'Spoke: {text}',
                        'text': text,
                        'language': language
                    })
                else:
                    self.status_callback({
                        'type': 'speech_error', 
                        'message': f'Failed to speak: {text}'
                    })
                    
        except Exception as e:
            print(f"❌ Speech thread error: {e}")
            if self.status_callback:
                self.status_callback({
                    'type': 'speech_error', 
                    'message': f'Speech error: {str(e)}'
                })
        
        finally:
            self.is_speaking = False

# Global instance
web_converter = SpeechToSign()

if __name__ == "__main__":
    # Test the converter
    converter = SpeechToSign()
    
    def test_callback(data):
        print(f"📝 Test Callback: {data}")
    
    def test_status(data):
        print(f"📊 Test Status: {data}")
    
    converter.set_callbacks(test_callback, test_status)
    
    # Test text to sign conversion
    test_text = "Hello World"
    sequence = converter.text_to_gloss_sequence(test_text)
    print(f"🔤 Test conversion: '{test_text}' -> {sequence}")
    
    # Test sign image generation
    print(f"🖼️ Sign images available: {list(converter.sign_images.keys())}")
    
    # Test speech synthesis
    print("🔊 Testing speech synthesis...")
    converter.speak_text("Hello, this is a test of the speech to sign system.")
    
    time.sleep(5)  # Wait for speech to complete