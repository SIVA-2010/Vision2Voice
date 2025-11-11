"""
COMPLETE ORIGINAL DETECTION SYSTEM - Fixed with C/Q sign detection
FIXED: C and Q letters are detected as signs when signed, but keyboard presses work separately
"""

import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import json
import time
import os
import sys
from collections import deque, Counter
import pyautogui
import threading
import tempfile
import pygame
from gtts import gTTS

# Add parent directory to path to import language_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from language_processor import language_processor
    print("✅ Language processor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import language_processor: {e}")
    # Create a mock language_processor for fallback
    class MockLanguageProcessor:
        def process_letter_sequence(self, text, target_lang=None):
            return {
                'raw_letters': text,
                'formatted_text': text.replace('SPACE', ' ').replace('ENTER', '. '),
                'corrected_text': text.replace('SPACE', ' ').replace('ENTER', '. '),
                'translated_text': text.replace('SPACE', ' ').replace('ENTER', '. '),
                'target_language': target_lang,
                'used_ai': False
            }
    language_processor = MockLanguageProcessor()

class SignLanguageDetector:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # Window dimensions
        self.window_width = 1280
        self.window_height = 720
        
        # Set correct paths relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load class mappings - FIXED PATH
        class_mapping_path = os.path.join(project_root, 'models', 'class_mapping.json')
        try:
            with open(class_mapping_path, 'r') as f:
                mapping = json.load(f)
                self.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
                print(f"📚 Loaded {len(self.idx_to_class)} classes: {list(self.idx_to_class.values())}")
        except FileNotFoundError:
            print(f"❌ Class mapping file not found at: {class_mapping_path}")
            raise
        
        # Set model path - FIXED PATH
        if model_path is None:
            model_path = os.path.join(project_root, 'models', 'sign_language_model_best.pth')
        
        # Load model with correct architecture
        self.model = self.create_model(len(self.idx_to_class))
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        self.model.eval()
        
        # Transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # ORIGINAL CAMERA INITIALIZATION - FIXED
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Could not open webcam. Trying alternative camera index...")
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print("❌ Could not open any webcam. Using dummy mode.")
                self.dummy_mode = True
            else:
                self.dummy_mode = False
        else:
            self.dummy_mode = False
        
        if not self.dummy_mode:
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)
        
        # Modern UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.box_color = (0, 255, 0)
        self.text_color = (255, 255, 255)
        self.highlight_color = (0, 165, 255)
        self.special_color = (255, 100, 0)
        self.accent_color = (148, 0, 211)
        self.success_color = (50, 205, 50)
        self.warning_color = (255, 165, 0)
        self.analyze_color = (255, 255, 0)
        self.background_color = (240, 240, 240)
        
        # Detection box
        self.box_size = 300
        self.box_margin = 20
        
        # Detection timing - ORIGINAL SETTINGS
        self.detection_cooldown = 2.0  # 2-second cooldown
        self.analysis_period = 2.0     # 2-second analysis
        self.is_in_cooldown = False
        self.is_analyzing = False
        self.cooldown_start_time = 0
        self.analysis_start_time = 0
        self.hand_in_box = False
        self.last_hand_state = False
        self.consecutive_detections = 0
        
        # Prediction history
        self.prediction_history = deque(maxlen=8)
        self.confidence_history = deque(maxlen=8)
        self.current_letter_sequence = ""
        self.last_prediction = ""
        self.detection_count = 0
        
        # Language processing state
        self.processed_output = ""
        self.formatted_text = ""
        self.corrected_text = ""
        self.target_language = None
        
        # TTS state
        self.is_speaking = False
        self.tts_queue = []
        
        # Frame processing
        self.frame_counter = 0
        self.process_every_n_frames = 2
        
        # Background for motion detection
        self.background = None
        self.background_initialized = False
        
        # Detection state for web interface
        self.detection_active = False
        self.current_detection_result = {
            'prediction': '',
            'confidence': 0.0,
            'current_text': '',
            'hand_detected': False,
            'is_analyzing': False,
            'is_in_cooldown': False
        }
        
        # Web interface callbacks
        self.update_callback = None
        self.status_callback = None
        self.message_callback = None
        
        # Manual key press tracking
        self.manual_key_pressed = False
        self.last_manual_key_time = 0
        self.manual_key_cooldown = 0.5  # 500ms cooldown for manual keys
        
        print("🎯 Detector initialized with original detection logic")

    def get_frame(self):
        """Get frame from webcam or create dummy frame"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame
            else:
                print("❌ Failed to read frame from webcam")
                return False, None
        elif self.dummy_mode:
            # Create a dummy frame for testing
            dummy_frame = np.ones((self.window_height, self.window_width, 3), dtype=np.uint8) * 128
            cv2.putText(dummy_frame, "DUMMY CAMERA MODE - TESTING", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(dummy_frame, "Press 'Q' to exit detection", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(dummy_frame, f"Text: {self.current_letter_sequence}", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return True, dummy_frame
        else:
            return False, None

    def set_callbacks(self, update_callback, status_callback, message_callback):
        """Set callbacks for web interface updates"""
        self.update_callback = update_callback
        self.status_callback = status_callback
        self.message_callback = message_callback

    def create_model(self, num_classes):
        """Create model that matches the training architecture"""
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except:
            try:
                model = models.resnet18(pretrained=True)
            except:
                model = models.resnet18(weights=None)
        
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(num_ftrs, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, num_classes)
        )
        
        return model.to(self.device)
    
    def is_hand_present(self, roi):
        """ORIGINAL HAND DETECTION FUNCTION"""
        if roi.size == 0:
            return False
        
        # In dummy mode, simulate hand detection
        if self.dummy_mode:
            # Simulate hand detection every 3 seconds for testing
            return time.time() % 6 < 3
        
        try:
            # Method 1: Skin color detection in HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / (roi.shape[0] * roi.shape[1])
            
            # Method 2: Motion detection
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if not self.background_initialized:
                self.background = gray
                self.background_initialized = True
                return False
            
            frame_diff = cv2.absdiff(self.background, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            motion_ratio = np.sum(thresh > 0) / (roi.shape[0] * roi.shape[1])
            
            self.background = cv2.addWeighted(self.background, 0.95, gray, 0.05, 0)
            
            hand_detected = (skin_ratio > 0.05) or (motion_ratio > 0.02)
            
            if hand_detected:
                self.consecutive_detections = min(self.consecutive_detections + 1, 10)
            else:
                self.consecutive_detections = max(self.consecutive_detections - 1, 0)
            
            return self.consecutive_detections >= 2
            
        except Exception as e:
            print(f"⚠️ Hand detection error: {e}")
            return False
    
    def predict(self, image):
        """ORIGINAL PREDICTION FUNCTION"""
        try:
            image = Image.fromarray(image).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                prediction = self.idx_to_class[predicted_idx.item()]
                confidence_value = confidence.item()
                
                return prediction, confidence_value
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "UNKNOWN", 0.0
         
    def process_language_output(self):
        """ORIGINAL LANGUAGE PROCESSING FUNCTION"""
        if not self.current_letter_sequence:
            print("⚠️ No text to process")
            return
        
        print(f"🔤 Processing: '{self.current_letter_sequence}'")
    
        try:
            result = language_processor.process_letter_sequence(
                self.current_letter_sequence, 
                self.target_language
            )
        
            # Store all processing stages for display
            self.formatted_text = result['formatted_text']
            self.corrected_text = result['corrected_text']
            self.processed_output = result['translated_text']
            
            print(f"🎯 Language Processing Complete:")
            print(f"   Raw: {result['raw_letters']}")
            print(f"   Formatted: {result['formatted_text']}")
            print(f"   Corrected: {result['corrected_text']}")
            if self.target_language:
                print(f"   Translated: {result['translated_text']}")
            print(f"   AI Powered: {result['used_ai']}")
            
            # Send update to web interface - SEND TO HARI ONLY
            if self.update_callback:
                self.update_callback({
                    'current_text': self.processed_output,
                    'raw_text': self.current_letter_sequence,
                    'processed_text': self.processed_output,
                    'formatted_text': self.formatted_text,
                    'corrected_text': self.corrected_text,
                    'timestamp': time.time()
                })
            
        except Exception as e:
            print(f"❌ Language processing error: {e}")
            # Fallback processing
            self.formatted_text = self.current_letter_sequence.replace('SPACE', ' ').replace('ENTER', '. ')
            self.corrected_text = self.formatted_text
            self.processed_output = self.formatted_text
  
    def handle_prediction(self, prediction, confidence):
        """UPDATED PREDICTION HANDLING - C and Q are now detected as signs"""
        print(f"🎯 Detected: {prediction} (confidence: {confidence:.3f})")
        
        # ALLOW C and Q as sign predictions - they will be visible in UI
        if prediction == 'BACKSPACE':
            if self.current_letter_sequence:
                if self.current_letter_sequence.endswith(('BACKSPACE', 'SPACE', 'ENTER')):
                    parts = self.current_letter_sequence.split()
                    if parts:
                        parts.pop()
                        self.current_letter_sequence = ' '.join(parts)
                else:
                    self.current_letter_sequence = self.current_letter_sequence[:-1]
                pyautogui.press('backspace')
                print("⌫ Backspace pressed")
                
        elif prediction == 'SPACE':
            self.current_letter_sequence += ' SPACE'
            pyautogui.write(' ')
            print("␣ Space added")
            
        elif prediction == 'ENTER':
            self.current_letter_sequence += ' ENTER'
            pyautogui.press('enter')
            print("↵ Enter pressed")
            
        else:
            # ALLOW ALL LETTERS including C and Q - they will be added to sequence
            self.current_letter_sequence += prediction
            pyautogui.write(prediction)
            print(f"✍️ Added '{prediction}' to sequence")
        
        # Auto-process after each detection
        self.process_language_output()

        self.detection_count += 1
        self.last_prediction = prediction
        
        # Start cooldown after successful detection
        self.is_in_cooldown = True
        self.cooldown_start_time = time.time()
        print(f"⏳ Cooldown started for {self.detection_cooldown} seconds")
    
    def create_gradient_background(self, width, height, color1, color2, horizontal=True):
        """Create a gradient background"""
        background = np.zeros((height, width, 3), dtype=np.uint8)
        if horizontal:
            for i in range(width):
                alpha = i / width
                color = [int(color1[j] * (1 - alpha) + color2[j] * alpha) for j in range(3)]
                background[:, i] = color
        else:
            for i in range(height):
                alpha = i / height
                color = [int(color1[j] * (1 - alpha) + color2[j] * alpha) for j in range(3)]
                background[i, :] = color
        return background
    
    def draw_ui(self, frame):
        """COMPLETE ORIGINAL UI"""
        h, w = frame.shape[:2]
        
        # Create modern gradient title background
        title_bg = self.create_gradient_background(w, 70, (25, 25, 112), (70, 130, 180))
        
        # Add stylish title with shadow
        title = "✨ Sign Language Recognition System ✨"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_COMPLEX, 0.9, 2)[0]
        title_x = (w - title_size[0]) // 2
        
        cv2.putText(title_bg, title, (title_x+2, 40+2), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(title_bg, title, (title_x, 40), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
        
        frame[0:70, 0:w] = title_bg
        
        # Left Panel - Text Information (60% width)
        left_panel_width = int(w * 0.6)
        left_panel_height = h - 70
        
        # Create left panel background
        left_panel = self.create_gradient_background(left_panel_width, left_panel_height, (30, 30, 30), (50, 50, 80))
        
        # Add section headers
        cv2.putText(left_panel, "📝 SIGN DETECTION", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2, cv2.LINE_AA)
        
        # Display current language mode
        lang_display = f"🌐 Language: {'English' if not self.target_language else 'Hindi' if self.target_language == 'hi' else 'Tamil'}"
        cv2.putText(left_panel, lang_display, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2, cv2.LINE_AA)
        
        # Display current letter sequence (for debugging only)
        seq_display = f"🔤 Detected: {self.current_letter_sequence}" if self.current_letter_sequence else "🔤 Start signing..."
        cv2.putText(left_panel, seq_display, (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
        
        # System Information Section
        cv2.putText(left_panel, "⚙️ SYSTEM INFO", (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2, cv2.LINE_AA)
        
        # Display mode info
        mode_info = "🤖 Powered by Gemini AI" if hasattr(language_processor, 'gemini_available') and language_processor.gemini_available else "📴 Enhanced Offline Processing"
        cv2.putText(left_panel, mode_info, (20, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1, cv2.LINE_AA)
        
        # Detection statistics
        stats_text = f"📈 Detections: {self.detection_count}"
        cv2.putText(left_panel, stats_text, (20, 205), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
        
        # TTS status
        tts_status = "🔊 SPEAKING..." if self.is_speaking else "🔇 Ready to speak"
        tts_color = (0, 255, 0) if not self.is_speaking else (0, 165, 255)
        cv2.putText(left_panel, tts_status, (20, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, tts_color, 2, cv2.LINE_AA)
        
        # Controls Section
        cv2.putText(left_panel, "🎮 CONTROLS", (20, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2, cv2.LINE_AA)
        
        controls = [
            "1 - Hindi | 2 - Tamil | 3 - English",
            "C - Clear | S - Speak | P - Process",
            "Q - Quit | R - Reset Detection"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(left_panel, control, (20, 300 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Place left panel on frame
        frame[70:70+left_panel_height, 0:left_panel_width] = left_panel
        
        # RIGHT PANEL - Camera Only (40% width)
        right_panel_start = left_panel_width
        right_panel_width = w - left_panel_width
        right_panel_height = h - 70
        
        # Extract the right portion of the camera frame
        camera_feed = frame[70:70+right_panel_height, right_panel_start:right_panel_start+right_panel_width]
        
        # Draw detection box in the center of right panel
        box_x1 = (right_panel_width - self.box_size) // 2
        box_y1 = (right_panel_height - self.box_size) // 2
        box_x2 = box_x1 + self.box_size
        box_y2 = box_y1 + self.box_size
        
        # Change box color based on state
        if self.is_analyzing:
            box_color = self.analyze_color  # YELLOW
            box_thickness = 4
        elif self.hand_in_box:
            box_color = self.box_color  # GREEN
            box_thickness = 3
        else:
            box_color = (0, 100, 0)  # DARK GREEN
            box_thickness = 2
        
        # Draw outer glow effect
        for i in range(1, 4):
            alpha = 1.0 - (i * 0.3)
            glow_color = tuple(int(c * alpha) for c in box_color)
            cv2.rectangle(camera_feed, (box_x1-i, box_y1-i), (box_x2+i, box_y2+i), glow_color, 1)
        
        # Draw main box
        cv2.rectangle(camera_feed, (box_x1, box_y1), (box_x2, box_y2), box_color, box_thickness)
        
        # Add animated corner markers
        pulse = int(5 * abs(np.sin(time.time() * 3))) + 10
        corner_color = self.accent_color
        
        cv2.line(camera_feed, (box_x1, box_y1), (box_x1 + pulse, box_y1), corner_color, 3)
        cv2.line(camera_feed, (box_x1, box_y1), (box_x1, box_y1 + pulse), corner_color, 3)
        cv2.line(camera_feed, (box_x2, box_y1), (box_x2 - pulse, box_y1), corner_color, 3)
        cv2.line(camera_feed, (box_x2, box_y1), (box_x2, box_y1 + pulse), corner_color, 3)
        cv2.line(camera_feed, (box_x1, box_y2), (box_x1 + pulse, box_y2), corner_color, 3)
        cv2.line(camera_feed, (box_x1, box_y2), (box_x1, box_y2 - pulse), corner_color, 3)
        cv2.line(camera_feed, (box_x2, box_y2), (box_x2 - pulse, box_y2), corner_color, 3)
        cv2.line(camera_feed, (box_x2, box_y2), (box_x2, box_y2 - pulse), corner_color, 3)
        
        # Add instruction text
        if self.is_analyzing:
            remaining_time = max(0, self.analysis_period - (time.time() - self.analysis_start_time))
            instruction_text = f"🔍 Analyzing: {remaining_time:.1f}s"
            text_color = self.analyze_color
        elif self.is_in_cooldown:
            remaining_time = max(0, self.detection_cooldown - (time.time() - self.cooldown_start_time))
            instruction_text = f"⏳ Cooldown: {remaining_time:.1f}s"
            text_color = self.highlight_color
        elif self.hand_in_box:
            instruction_text = "✅ Hand Detected - Hold Still!"
            text_color = self.success_color
        else:
            instruction_text = "👋 Place Hand Here"
            text_color = (0, 255, 255)
        
        text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = (right_panel_width - text_size[0]) // 2
        text_y = box_y1 - 20
        
        cv2.putText(camera_feed, instruction_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
        
        # Add detection state at bottom
        state_text = f"🔒 State: {'ANALYZING' if self.is_analyzing else 'COOLDOWN' if self.is_in_cooldown else 'READY'}"
        state_size = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        state_x = (right_panel_width - state_size[0]) // 2
        state_y = right_panel_height - 20
        
        cv2.putText(camera_feed, state_text, (state_x, state_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)
        
        # Place the modified camera feed back to frame
        frame[70:70+right_panel_height, right_panel_start:right_panel_start+right_panel_width] = camera_feed
        
        return box_x1 + right_panel_start, box_y1 + 70, box_x2 + right_panel_start, box_y2 + 70
    
    def enhanced_speech_synthesis(self, text, language='en'):
        """Enhanced speech synthesis using multiple methods"""
        try:
            print(f"🔊 Speaking: '{text}' in {language}")
            
            # Method 1: Try pyttsx3 first (works offline)
            try:
                import pyttsx3
                engine = pyttsx3.init()
                
                # Set language properties
                if language == 'hi':  # Hindi
                    # Try to set Hindi voice if available
                    voices = engine.getProperty('voices')
                    for voice in voices:
                        if 'hindi' in voice.name.lower() or 'india' in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
                elif language == 'ta':  # Tamil
                    # Try to set Tamil voice if available
                    voices = engine.getProperty('voices')
                    for voice in voices:
                        if 'tamil' in voice.name.lower() or 'india' in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
                
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.8)
                
                engine.say(text)
                engine.runAndWait()
                print("✅ Speech completed with pyttsx3")
                return True
                
            except Exception as pyttsx_error:
                print(f"⚠️ pyttsx3 failed: {pyttsx_error}")
                
            # Method 2: Try gTTS as fallback
            try:
                from gtts import gTTS
                import pygame
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tts = gTTS(text=text, lang=language, slow=False)
                    tts.save(tmp_file.name)
                    
                    # Play audio
                    pygame.mixer.init()
                    pygame.mixer.music.load(tmp_file.name)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to complete
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    # Cleanup
                    pygame.mixer.quit()
                    os.unlink(tmp_file.name)
                
                print("✅ Speech completed with gTTS")
                return True
                
            except Exception as gtts_error:
                print(f"⚠️ gTTS failed: {gtts_error}")
            
            # Method 3: Platform-specific solutions
            try:
                import platform
                system = platform.system()
                
                if system == "Darwin":  # macOS
                    os.system(f'say "{text}"')
                    print("✅ Speech completed with macOS say")
                    return True
                elif system == "Linux":  # Linux
                    os.system(f'espeak "{text}"')
                    print("✅ Speech completed with espeak")
                    return True
                elif system == "Windows":  # Windows
                    import win32com.client
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    speaker.Speak(text)
                    print("✅ Speech completed with Windows SAPI")
                    return True
                    
            except Exception as platform_error:
                print(f"⚠️ Platform speech failed: {platform_error}")
            
            print("❌ All speech synthesis methods failed")
            return False
                
        except Exception as e:
            print(f"❌ Enhanced speech error: {e}")
            return False

    def speak_text(self, text=None):
        """Enhanced speech with multiple fallback methods"""
        if not text:
            text = self.processed_output if self.processed_output else self.current_letter_sequence
            
        if not text or text.isspace():
            print("⚠️ No text to speak")
            return
        
        try:
            text = text.strip()
            
            # Determine language
            lang = 'en'
            if self.target_language == 'hi':
                lang = 'hi'
            elif self.target_language == 'ta':
                lang = 'ta'
            
            # Use enhanced speech synthesis in a separate thread
            def speech_thread():
                self.is_speaking = True
                success = self.enhanced_speech_synthesis(text, lang)
                if not success:
                    print("❌ Speech synthesis failed - no audio output")
                self.is_speaking = False
            
            threading.Thread(target=speech_thread, daemon=True).start()
            print(f"🔊 Started enhanced speech: '{text}'")
                
        except Exception as e:
            print(f"❌ Speech error: {e}")
    
    def run_detection(self):
        """COMPLETE ORIGINAL MAIN LOOP - Only runs when triggered"""
        print("🚀 Starting Sign Language Detection System...")
        print("=" * 70)
        print("📝 SYSTEM STATUS:")
        if hasattr(language_processor, 'gemini_available') and language_processor.gemini_available:
            print("   • AI: 🤖 Gemini AI Active")
        else:
            print("   • AI: 📴 Gemini AI Not Available")
        print("   • Translation: Hindi & Tamil Available")
        print("   • TTS: Enhanced Multi-language Speech")
        print("   • UI: Full Feature Display Enabled")
        print("=" * 70)
        print("Controls: 1-Hindi, 2-Tamil, 3-English, C-Clear, S-Speak, P-Process, Q-Quit, R-Reset")
        print("=" * 70)
        
        cv2.namedWindow('Sign Language System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sign Language System', self.window_width, self.window_height)
        
        prev_time = time.time()
        fps = 0
        
        print("📹 Starting webcam feed...")
        
        self.detection_active = True
        
        # Send initial status
        if self.status_callback:
            self.status_callback({
                'detection_active': True,
                'hand_detected': False,
                'is_analyzing': False,
                'is_in_cooldown': False,
                'last_prediction': '',
                'detection_count': self.detection_count
            })
        
        try:
            while self.detection_active:
                # Get frame from webcam or dummy
                ret, frame = self.get_frame()
                if not ret:
                    print("❌ Failed to get frame")
                    time.sleep(0.1)
                    continue
                    
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (self.window_width, self.window_height))
                
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time))
                prev_time = curr_time
                
                # Get box coordinates from UI
                box_x1, box_y1, box_x2, box_y2 = self.draw_ui(frame)
                
                # Extract ROI for prediction
                roi = frame[box_y1:box_y2, box_x1:box_x2]
                
                # Detect hand presence - ORIGINAL FUNCTION
                current_hand_state = self.is_hand_present(roi) if roi.size > 0 else False
                
                current_time = time.time()
                
                # Check if hand just entered the box - ORIGINAL LOGIC
                if current_hand_state and not self.last_hand_state and not self.is_in_cooldown:
                    self.is_analyzing = True
                    self.analysis_start_time = current_time
                    self.prediction_history.clear()
                    self.confidence_history.clear()
                    print("🔄 Hand detected! Starting 2-second analysis period...")
                
                self.hand_in_box = current_hand_state
                self.last_hand_state = current_hand_state
                
                prediction = ""
                confidence = 0.0
                
                self.frame_counter += 1
                should_process = (self.frame_counter % self.process_every_n_frames == 0)
                
                # ORIGINAL DETECTION LOGIC - Only during analysis period
                if (self.is_analyzing and 
                    self.hand_in_box and 
                    roi.size > 0 and 
                    should_process):
                    
                    try:
                        roi_resized = cv2.resize(roi, (128, 128))
                        pred, conf = self.predict(roi_resized)
                        
                        if conf > 0.3:
                            self.prediction_history.append(pred)
                            self.confidence_history.append(conf)
                            prediction = pred
                            confidence = conf
                            
                            print(f"📊 Frame {self.frame_counter}: {pred} ({conf:.3f})")
                            
                    except Exception as e:
                        print(f"❌ Frame processing error: {e}")
                
                # Check if analysis period is complete - ORIGINAL TIMING
                if self.is_analyzing and (current_time - self.analysis_start_time >= self.analysis_period):
                    self.is_analyzing = False
                    
                    if len(self.prediction_history) >= 3:
                        most_common = Counter(self.prediction_history).most_common(1)
                        if most_common:
                            final_prediction = most_common[0][0]
                            avg_confidence = np.mean(self.confidence_history)
                            
                            if avg_confidence > 0.5:
                                print(f"🎯 Final Prediction: {final_prediction} (confidence: {avg_confidence:.3f})")
                                self.handle_prediction(final_prediction, avg_confidence)
                            else:
                                print(f"⚠️ Low confidence prediction: {final_prediction} ({avg_confidence:.3f})")
                        else:
                            print("❌ No consistent prediction during analysis")
                    else:
                        print("❌ Insufficient predictions during analysis")
                
                # Check if cooldown period is complete - ORIGINAL TIMING
                if self.is_in_cooldown and (current_time - self.cooldown_start_time >= self.detection_cooldown):
                    self.is_in_cooldown = False
                    print("🔄 Cooldown complete - Ready for next detection")
                
                # Update current detection result for web interface
                self.current_detection_result = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'current_text': self.processed_output if self.processed_output else self.current_letter_sequence,
                    'hand_detected': self.hand_in_box,
                    'is_analyzing': self.is_analyzing,
                    'is_in_cooldown': self.is_in_cooldown
                }
                
                # Send status update to web interface
                if self.status_callback:
                    self.status_callback({
                        'detection_active': True,
                        'hand_detected': self.hand_in_box,
                        'is_analyzing': self.is_analyzing,
                        'is_in_cooldown': self.is_in_cooldown,
                        'last_prediction': prediction,
                        'detection_count': self.detection_count
                    })
                
                # Display FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, self.window_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show the frame
                cv2.imshow('Sign Language System', frame)
                
                # Handle key presses - ORIGINAL CONTROLS
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Quitting detection system...")
                    break
                elif key == ord('c'):
                    print("🗑️ Clearing text...")
                    self.current_letter_sequence = ""
                    self.processed_output = ""
                    self.formatted_text = ""
                    self.corrected_text = ""
                elif key == ord('s'):
                    print("🔊 Speaking text...")
                    self.speak_text()
                elif key == ord('p'):
                    print("🔄 Processing text...")
                    self.process_language_output()
                elif key == ord('1'):
                    print("🌐 Switching to Hindi")
                    self.target_language = 'hi'
                    self.process_language_output()
                elif key == ord('2'):
                    print("🌐 Switching to Tamil")
                    self.target_language = 'ta'
                    self.process_language_output()
                elif key == ord('3'):
                    print("🌐 Switching to English")
                    self.target_language = None
                    self.process_language_output()
                elif key == ord('r'):
                    print("🔄 Resetting detection state...")
                    self.is_analyzing = False
                    self.is_in_cooldown = False
                    self.prediction_history.clear()
                    self.confidence_history.clear()
                
        except KeyboardInterrupt:
            print("\n🛑 Detection interrupted by user")
        except Exception as e:
            print(f"❌ Detection error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.detection_active = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")

def main():
    """Main function"""
    try:
        detector = SignLanguageDetector()
        detector.run_detection()
    except Exception as e:
        print(f"❌ Failed to start detection: {e}")
        print("💡 Make sure:")
        print("   • Webcam is connected")
        print("   • Model files exist in models/ directory")
        print("   • Dependencies are installed")
        print("   • Class mapping file is present")

if __name__ == "__main__":
    main()