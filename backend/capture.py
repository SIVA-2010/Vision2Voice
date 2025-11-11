import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class SignLanguageDataCollector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create data directory
        self.data_dir = "data/train"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Letters to capture (A-Z + Special buttons)
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                       'BACKSPACE', 'SPACE', 'ENTER']
        
        # Create directories and initialize counters
        self.letter_counters = {}
        for letter in self.letters:
            letter_dir = os.path.join(self.data_dir, letter)
            os.makedirs(letter_dir, exist_ok=True)
            # Count existing images to continue from last number
            self.letter_counters[letter] = self.count_existing_images(letter_dir)
        
        # Current letter and counters
        self.current_letter = 'A'
        self.total_images = 300  # Images per letter
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.box_color = (0, 255, 0)
        self.text_color = (255, 255, 255)
        self.highlight_color = (0, 165, 255)
        self.special_color = (255, 100, 0)
        
        # Detection box
        self.box_size = 300
        self.box_margin = 40
        
    def count_existing_images(self, directory):
        """Count existing images in directory to continue numbering"""
        if not os.path.exists(directory):
            return 0
            
        image_files = [f for f in os.listdir(directory) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            return 0
        
        # Extract numbers from filenames and find the maximum
        numbers = []
        for filename in image_files:
            # Remove extension and split by underscore
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                numbers.append(int(parts[-1]))
        
        return max(numbers) + 1 if numbers else 0
    
    def draw_ui(self, frame):
        h, w = frame.shape[:2]
        
        # Create title background
        title_bg = np.zeros((80, w, 3), dtype=np.uint8)
        title_bg[:,:] = (25, 25, 25)
        
        # Add title
        title = "Sign Language Data Collector (A-Z + Special Buttons)"
        title_size = cv2.getTextSize(title, self.font, 0.8, 2)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(title_bg, title, (title_x, 50), self.font, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        
        # Add title bar to frame
        frame[0:80, 0:w] = title_bg
        
        # Draw detection box (center-right)
        box_x2 = w - self.box_margin
        box_x1 = box_x2 - self.box_size
        box_y1 = (h - self.box_size) // 2
        box_y2 = box_y1 + self.box_size
        
        # Draw fancy box
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), self.box_color, 2)
        
        # Add corner markers
        corner_size = 15
        # Top-left
        cv2.line(frame, (box_x1, box_y1), (box_x1 + corner_size, box_y1), self.box_color, 2)
        cv2.line(frame, (box_x1, box_y1), (box_x1, box_y1 + corner_size), self.box_color, 2)
        # Top-right
        cv2.line(frame, (box_x2, box_y1), (box_x2 - corner_size, box_y1), self.box_color, 2)
        cv2.line(frame, (box_x2, box_y1), (box_x2, box_y1 + corner_size), self.box_color, 2)
        # Bottom-left
        cv2.line(frame, (box_x1, box_y2), (box_x1 + corner_size, box_y2), self.box_color, 2)
        cv2.line(frame, (box_x1, box_y2), (box_x1, box_y2 - corner_size), self.box_color, 2)
        # Bottom-right
        cv2.line(frame, (box_x2, box_y2), (box_x2 - corner_size, box_y2), self.box_color, 2)
        cv2.line(frame, (box_x2, box_y2), (box_x2, box_y2 - corner_size), self.box_color, 2)
        
        # Add instruction text
        cv2.putText(frame, "Place hand inside box", (box_x1, box_y1 - 15), 
                   self.font, 0.7, self.box_color, 2, cv2.LINE_AA)
        
        # Create status bar at bottom
        status_bar = np.zeros((120, w, 3), dtype=np.uint8)
        status_bar[:,:] = (40, 40, 40)
        
        # Add status information
        current_count = self.letter_counters[self.current_letter]
        status_text = f"Current: {self.current_letter} | Captured: {current_count}/{self.total_images}"
        cv2.putText(status_bar, status_text, (20, 40), self.font, 0.8, self.text_color, 2, cv2.LINE_AA)
        
        # Add controls info
        controls_text = "Press A-Z,1,2,3: Change gesture | SPACE: Capture | ESC: Quit"
        cv2.putText(status_bar, controls_text, (20, 75), self.font, 0.6, self.highlight_color, 1, cv2.LINE_AA)
        
        # Add special buttons info
        special_text = "1: BACKSPACE, 2: SPACE, 3: ENTER"
        cv2.putText(status_bar, special_text, (20, 100), self.font, 0.6, self.special_color, 1, cv2.LINE_AA)
        
        # Add status bar to frame
        frame[h-120:h, 0:w] = status_bar
        
        # Add letter buttons (show first 10 items for space)
        visible_items = self.letters[:10]
        button_width = w // len(visible_items)
        for i, item in enumerate(visible_items):
            button_x1 = i * button_width
            button_x2 = (i + 1) * button_width
            
            # Highlight current item with different colors
            if item == self.current_letter:
                color = self.highlight_color
            elif item in ['BACKSPACE', 'SPACE', 'ENTER']:
                color = self.special_color
            else:
                color = (100, 100, 100)
                
            cv2.rectangle(frame, (button_x1, 80), (button_x2, 130), color, -1)
            
            # Add item text (shorten special buttons for display)
            display_text = item[:3] if item in ['BACKSPACE', 'SPACE', 'ENTER'] else item
            text_size = cv2.getTextSize(display_text, self.font, 0.6, 1)[0]
            text_x = button_x1 + (button_width - text_size[0]) // 2
            text_y = 110
            cv2.putText(frame, display_text, (text_x, text_y), self.font, 0.6, self.text_color, 1, cv2.LINE_AA)
        
        return box_x1, box_y1, box_x2, box_y2
    
    def run(self):
        print("Starting data collection...")
        print("Instructions:")
        print("- Press A-Z to select letters")
        print("- Press 1 for BACKSPACE, 2 for SPACE, 3 for ENTER")
        print("- Press SPACE to capture an image")
        print("- Press ESC to quit")
        print(f"Available gestures: {', '.join(self.letters)}")
        
        # Show initial counters
        print("\nCurrent image counts per letter:")
        for letter in self.letters:
            print(f"  {letter}: {self.letter_counters[letter]} images")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            
            # Draw UI and get box coordinates
            box_x1, box_y1, box_x2, box_y2 = self.draw_ui(frame)
            
            # Display the frame
            cv2.imshow('Sign Language Data Collector', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to quit (changed from 'q')
                break
            elif key >= ord('a') and key <= ord('z'):
                letter = chr(key).upper()
                if letter in self.letters:
                    self.current_letter = letter
                    current_count = self.letter_counters[self.current_letter]
                    print(f"Switched to letter: {self.current_letter} (Current count: {current_count})")
            elif key == ord('1'):
                if 'BACKSPACE' in self.letters:
                    self.current_letter = 'BACKSPACE'
                    current_count = self.letter_counters[self.current_letter]
                    print(f"Switched to BACKSPACE gesture (Current count: {current_count})")
            elif key == ord('2'):
                if 'SPACE' in self.letters:
                    self.current_letter = 'SPACE'
                    current_count = self.letter_counters[self.current_letter]
                    print(f"Switched to SPACE gesture (Current count: {current_count})")
            elif key == ord('3'):
                if 'ENTER' in self.letters:
                    self.current_letter = 'ENTER'
                    current_count = self.letter_counters[self.current_letter]
                    print(f"Switched to ENTER gesture (Current count: {current_count})")
            elif key == ord(' '):  # Space to capture
                current_count = self.letter_counters[self.current_letter]
                if current_count < self.total_images:
                    # Extract and save ROI
                    roi = frame[box_y1:box_y2, box_x1:box_x2]
                    if roi.size > 0:
                        # Resize to consistent size
                        roi = cv2.resize(roi, (128, 128))
                        
                        # Save image with proper numbering
                        filename = f"{self.data_dir}/{self.current_letter}/{self.current_letter}_{current_count:04d}.jpg"
                        cv2.imwrite(filename, roi)
                        
                        # Update counter
                        self.letter_counters[self.current_letter] += 1
                        new_count = self.letter_counters[self.current_letter]
                        
                        if new_count % 50 == 0:
                            print(f"✅ Saved {filename} | Total for {self.current_letter}: {new_count}")
                        elif new_count % 10 == 0:
                            print(f"📸 Captured image {new_count} for {self.current_letter}")
                        
                        # Visual feedback
                        flash = frame.copy()
                        cv2.rectangle(flash, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
                        cv2.imshow('Sign Language Data Collector', flash)
                        cv2.waitKey(50)
                else:
                    print(f"🎯 Reached maximum images ({self.total_images}) for {self.current_letter}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final summary
        print("\n📊 Data Collection Summary:")
        for letter in self.letters:
            count = self.letter_counters[letter]
            print(f"  {letter}: {count} images")

if __name__ == "__main__":
    collector = SignLanguageDataCollector()
    collector.run()