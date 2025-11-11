"""
COMPLETE FLASK BACKEND WITH ENHANCED MULTILINGUAL SUPPORT
FIXED: Gemini API integration for grammar correction and translation
FIXED: Enhanced Hindi and Tamil translation support
FIXED: Improved speech synthesis and sign image serving
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import sys
import logging
import threading
import time
import json
from datetime import datetime
import base64
import cv2
import numpy as np
import glob

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app with correct paths
frontend_dir = os.path.join(project_root, 'frontend')

app = Flask(__name__, 
           static_folder=frontend_dir,
           template_folder=frontend_dir)
app.config['SECRET_KEY'] = 'sign-language-chat-secret'
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode='threading',
                   logger=True,
                   engineio_logger=True)
CORS(app)

# Import modules with proper error handling
print("🚀 Initializing modules...")

# Import detector class and create instance
try:
    from backend.detect import SignLanguageDetector
    detector = SignLanguageDetector()
    print("✅ Detector imported and initialized successfully")
except ImportError as e:
    print(f"❌ Failed to import detector: {e}")
    detector = None
except Exception as e:
    print(f"❌ Failed to initialize detector: {e}")
    detector = None

try:
    from backend.language_processor import language_processor
    print("✅ Language processor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import language_processor: {e}")
    language_processor = None

try:
    from backend.chat_manager import chat_manager
    print("✅ Chat manager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import chat_manager: {e}")
    chat_manager = None

try:
    from backend.speech_to_sign import web_converter
    print("✅ Speech to sign converter imported successfully")
except ImportError as e:
    print(f"❌ Failed to import web_converter: {e}")
    web_converter = None

# Global state
active_users = {
    'siva': {'connected': False, 'sid': None, 'current_text': ''},
    'hari': {'connected': False, 'sid': None, 'current_text': ''}
}

current_detection_sid = None

# Load sign images from dataset
sign_images = {}
def load_sign_images():
    """Load actual sign images from dataset"""
    print("🖼️ Loading sign images from dataset...")
    data_dir = os.path.join(project_root, 'data', 'train')
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        return
    
    # Define all possible sign classes
    sign_classes = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'SPACE', 'ENTER', 'BACKSPACE'
    ]
    
    for sign_class in sign_classes:
        if sign_class in ['SPACE', 'ENTER', 'BACKSPACE']:
            # Create placeholder for special characters
            sign_images[sign_class] = create_placeholder_image(sign_class)
            continue
            
        # Look for image files for this class
        class_dir = os.path.join(data_dir, sign_class)
        if os.path.exists(class_dir):
            image_files = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                         glob.glob(os.path.join(class_dir, '*.jpeg')) + \
                         glob.glob(os.path.join(class_dir, '*.png'))
            
            if image_files:
                # Load the first image found
                try:
                    img = cv2.imread(image_files[0])
                    if img is not None:
                        # Resize to standard size
                        img = cv2.resize(img, (200, 200))
                        sign_images[sign_class] = img
                        print(f"✅ Loaded image for {sign_class}: {os.path.basename(image_files[0])}")
                    else:
                        print(f"❌ Failed to load image for {sign_class}")
                        sign_images[sign_class] = create_placeholder_image(sign_class)
                except Exception as e:
                    print(f"❌ Error loading image for {sign_class}: {e}")
                    sign_images[sign_class] = create_placeholder_image(sign_class)
            else:
                print(f"⚠️ No images found for {sign_class}, creating placeholder")
                sign_images[sign_class] = create_placeholder_image(sign_class)
        else:
            print(f"⚠️ Directory not found for {sign_class}, creating placeholder")
            sign_images[sign_class] = create_placeholder_image(sign_class)
    
    print(f"✅ Loaded {len(sign_images)} sign images")

def create_placeholder_image(sign_class):
    """Create a placeholder image when actual image is not available"""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
    
    if sign_class == 'SPACE':
        cv2.rectangle(img, (50, 80), (150, 120), (0, 0, 255), 2)
        cv2.putText(img, "SPACE", (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif sign_class == 'ENTER':
        cv2.rectangle(img, (50, 80), (150, 120), (0, 128, 0), 2)
        cv2.putText(img, "ENTER", (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 2)
    elif sign_class == 'BACKSPACE':
        cv2.rectangle(img, (50, 80), (150, 120), (255, 0, 0), 2)
        cv2.putText(img, "BACKSPACE", (40, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    else:
        # Letter placeholder
        color = (100, 100, 200)  # Blueish color for letters
        cv2.rectangle(img, (10, 10), (190, 190), color, -1)
        cv2.rectangle(img, (10, 10), (190, 190), (0, 0, 0), 2)
        
        # Add letter text
        text_size = cv2.getTextSize(sign_class, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = (200 - text_size[0]) // 2
        text_y = (200 + text_size[1]) // 2
        cv2.putText(img, sign_class, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    return img

# Load sign images on startup
load_sign_images()

# Web interface callbacks
def send_detection_update(data):
    """Send detection update to specific client"""
    if current_detection_sid:
        socketio.emit('detection_update', data, room=current_detection_sid)
    
    # Send to Hari in real-time
    hari_sid = active_users['hari']['sid']
    if hari_sid and data.get('processed_text'):
        logger.info(f"📤 REAL-TIME: Sending to Hari: {data['processed_text']}")
        socketio.emit('receive_siva_message', {
            'from': 'siva',
            'message': data['processed_text'],
            'type': 'sign_text',
            'timestamp': datetime.now().isoformat(),
            'original_text': data.get('original_text', ''),
            'corrected_text': data.get('corrected_text', ''),
            'processed_text': data['processed_text']
        }, room=hari_sid)

def send_detection_status(data):
    """Send detection status to specific client"""
    if current_detection_sid:
        socketio.emit('detection_status', data, room=current_detection_sid)

def send_detection_error(error_msg):
    """Send detection error to specific client"""
    if current_detection_sid:
        socketio.emit('detection_error', {'error': error_msg}, room=current_detection_sid)

def send_message_to_hari(data):
    """Send message from Siva to Hari"""
    hari_sid = active_users['hari']['sid']
    if hari_sid and data['message']:
        logger.info(f"📤 MANUAL: Sending message to Hari: {data['message']}")
        
        # Send to Hari
        socketio.emit('receive_siva_message', {
            'from': 'siva',
            'message': data['message'],
            'type': 'sign_text',
            'timestamp': datetime.now().isoformat(),
            'original_text': data.get('original_text', ''),
            'corrected_text': data.get('corrected_text', ''),
            'processed_text': data['message']
        }, room=hari_sid)
        
        # Also add to chat history
        if chat_manager:
            chat_manager.add_message('siva_hari_chat', 'siva', data['message'])
        
        logger.info(f"💬 Message sent to Hari: {data['message']}")

# Serve frontend files
@app.route('/')
def index():
    """Serve the main landing page"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/siva')
def siva():
    """Serve Siva's interface"""
    return send_from_directory(app.static_folder, 'siva.html')

@app.route('/hari')
def hari():
    """Serve Hari's interface"""
    return send_from_directory(app.static_folder, 'hari.html')

# Serve static files
@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files"""
    css_dir = os.path.join(app.static_folder, 'css')
    return send_from_directory(css_dir, filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    """Serve JavaScript files"""
    js_dir = os.path.join(app.static_folder, 'js')
    return send_from_directory(js_dir, filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve asset files"""
    assets_dir = os.path.join(app.static_folder, 'assets')
    return send_from_directory(assets_dir, filename)

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'running',
        'detector_ready': detector is not None,
        'language_processor_ready': language_processor is not None,
        'chat_manager_ready': chat_manager is not None,
        'speech_ready': web_converter is not None,
        'sign_images_loaded': len(sign_images),
        'active_users': {k: v['connected'] for k, v in active_users.items()},
        'detection_active': detector.detection_active if detector else False,
        'gemini_available': language_processor.gemini_available if language_processor else False
    })

@app.route('/api/sign-image/<sign_class>')
def serve_sign_image(sign_class):
    """Serve sign images from dataset"""
    try:
        logger.info(f"🖼️ Requesting sign image for: {sign_class}")
        
        if sign_class in sign_images:
            img = sign_images[sign_class]
            
            # Convert numpy array to JPEG
            success, buffer = cv2.imencode('.jpg', img)
            if success:
                response = app.response_class(
                    response=buffer.tobytes(),
                    status=200,
                    mimetype='image/jpeg'
                )
                logger.info(f"✅ Successfully served image for: {sign_class}")
                return response
        
        # Create a placeholder image if sign not found
        logger.info(f"⚠️ Sign image not found for: {sign_class}, creating placeholder")
        placeholder = create_placeholder_image(sign_class)
        success, buffer = cv2.imencode('.jpg', placeholder)
        if success:
            response = app.response_class(
                response=buffer.tobytes(),
                status=200,
                mimetype='image/jpeg'
            )
            return response
            
        return jsonify({'error': 'Sign image not found'}), 404
        
    except Exception as e:
        logger.error(f"Error serving sign image {sign_class}: {e}")
        # Return a simple error image
        error_img = create_error_image(str(e))
        success, buffer = cv2.imencode('.jpg', error_img)
        if success:
            response = app.response_class(
                response=buffer.tobytes(),
                status=200,
                mimetype='image/jpeg'
            )
            return response
        return jsonify({'error': 'Internal server error'}), 500
@app.route('/api/language-status')
def api_language_status():
    """API endpoint to check language processor status"""
    if language_processor:
        status = language_processor.get_status()
        return jsonify(status)
    else:
        return jsonify({
            'gemini_available': False,
            'available_models': [],
            'current_model': None,
            'error': 'Language processor not available'
        })

def create_error_image(error_msg):
    """Create an error image"""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
    cv2.putText(img, "ERROR", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Loading Image", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connection_status', {'status': 'connected', 'message': 'Successfully connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    global current_detection_sid
    logger.info(f"Client disconnected: {request.sid}")
    
    # Stop detection if this user was running it
    if current_detection_sid == request.sid and detector:
        detector.stop_detection()
        current_detection_sid = None
    
    # Remove from active users
    for user, data in active_users.items():
        if data['sid'] == request.sid:
            active_users[user]['connected'] = False
            active_users[user]['sid'] = None
            logger.info(f"User {user} disconnected")
            
            # Leave room through chat manager
            if chat_manager:
                chat_manager.leave_room('siva_hari_chat', user)
            
            # Notify other users
            emit('user_status', {
                'user': user,
                'status': 'disconnected',
                'message': f'{user.capitalize()} has left the chat'
            }, broadcast=True, include_self=False)
            break

@socketio.on('join_chat')
def handle_join_chat(data):
    user_type = data.get('user_type')
    room_id = data.get('room_id', 'siva_hari_chat')
    
    if user_type in ['siva', 'hari']:
        active_users[user_type]['connected'] = True
        active_users[user_type]['sid'] = request.sid
        
        logger.info(f"User {user_type} joined room {room_id} with SID: {request.sid}")
        
        # Join room through chat manager
        if chat_manager:
            chat_manager.join_room(room_id, user_type, request.sid)
        
        emit('join_status', {
            'status': 'joined', 
            'room_id': room_id,
            'user_type': user_type
        })
        
        # Send chat history
        if chat_manager:
            history = chat_manager.get_chat_history(room_id)
            emit('chat_history', history)
        
        # Notify other users
        emit('user_status', {
            'user': user_type,
            'status': 'connected',
            'message': f'{user_type.capitalize()} has joined the chat'
        }, broadcast=True)

@socketio.on('start_opencv_detection')
def handle_start_detection():
    """Start the original detection system ONLY when triggered"""
    global current_detection_sid
    
    if detector is None:
        emit('detection_error', {'error': 'Detection system not available'})
        return
    
    if detector.detection_active:
        emit('detection_error', {'error': 'Detection already running'})
        return
    
    # Check if user is Siva
    user_type = None
    for user, user_data in active_users.items():
        if user_data['sid'] == request.sid:
            user_type = user
            break
    
    if user_type != 'siva':
        emit('detection_error', {'error': 'Only Siva can start detection'})
        return
    
    current_detection_sid = request.sid
    
    try:
        # Set up callbacks for web interface updates
        detector.set_callbacks(send_detection_update, send_detection_status, send_message_to_hari)
        
        # Start detection in a separate thread - ONLY WHEN TRIGGERED
        detector.detection_active = True
        detection_thread = threading.Thread(target=detector.run_detection, daemon=True)
        detection_thread.start()
        
        logger.info("🎬 Original detection system started on demand")
        emit('detection_started', {
            'message': 'Original detection system started successfully! OpenCV window should open.',
            'timestamp': time.time()
        })
            
    except Exception as e:
        logger.error(f"❌ Failed to start detection: {e}")
        emit('detection_error', {'error': f'Failed to start detection: {str(e)}'})

@socketio.on('stop_opencv_detection')
def handle_stop_detection():
    """Stop the detection system"""
    global current_detection_sid
    
    if detector:
        detector.detection_active = False
        current_detection_sid = None
        emit('detection_stopped', {'message': 'Detection stopped'})
        logger.info("🛑 Detection system stopped")

@socketio.on('clear_text')
def handle_clear_text():
    """Clear the current text (C key equivalent)"""
    if detector:
        detector.clear_text()
        emit('text_cleared', {'message': 'Text cleared'})
        logger.info("🗑️ Text cleared via web interface")

@socketio.on('reset_detection')
def handle_reset_detection():
    """Reset detection state (R key equivalent)"""
    if detector:
        detector.reset_detection()
        emit('detection_reset', {'message': 'Detection reset'})
        logger.info("🔄 Detection reset via web interface")

@socketio.on('set_language')
def handle_set_language(data):
    """Set target language for translation"""
    language = data.get('language')
    if detector:
        detector.set_language(language)
        emit('language_set', {'language': language})
        logger.info(f"🌐 Language set to: {language}")

@socketio.on('process_text')
def handle_process_text():
    """Process text through language processor (P key equivalent)"""
    if detector:
        detector.process_language_output()
        emit('text_processed', {'message': 'Text processed'})
        logger.info("🔤 Text processed via web interface")

@socketio.on('speak_text')
def handle_speak_text():
    """Speak the current text (S key equivalent)"""
    if detector:
        detector.speak_text()
        emit('speech_started', {'message': 'Speech started'})
        logger.info("🔊 Speech started via web interface")

@socketio.on('send_message_to_hari')
def handle_send_message_to_hari():
    """Send current processed text to Hari"""
    if detector:
        detector.send_message_to_hari()
        emit('message_sent', {'message': 'Message sent to Hari'})
        logger.info("💬 Message sent to Hari via web interface")

@socketio.on('send_chat_message')
def handle_send_chat_message(data):
    """Handle chat message sending from both users - ENHANCED GRAMMAR CORRECTION"""
    message = data.get('message', '').strip()
    user_type = data.get('user_type')
    target_language = data.get('target_language', 'en')
    
    if not message or user_type not in ['siva', 'hari']:
        emit('message_error', {'error': 'Invalid message or user'})
        return
    
    # ENHANCED: Improved grammar correction with better feedback
    processed_message = message
    corrected_text = message
    formatted_text = message
    used_ai = False
    
    if language_processor:
        try:
            # Process through language processor for grammar correction
            logger.info(f"🔤 Processing message with language processor: '{message}'")
            result = language_processor.process_letter_sequence(message, target_language)
            
            processed_message = result['translated_text'] if target_language != 'en' else result['corrected_text']
            corrected_text = result['corrected_text']
            formatted_text = result['formatted_text']
            used_ai = result['used_ai']
            
            logger.info(f"✅ Language Processing Result:")
            logger.info(f"   Original: '{message}'")
            logger.info(f"   Corrected: '{corrected_text}'")
            logger.info(f"   Formatted: '{formatted_text}'")
            logger.info(f"   Translated: '{result['translated_text']}'")
            logger.info(f"   Used AI: {used_ai}")
            
            # Send grammar correction info to the sender if correction occurred
            if corrected_text != message:
                emit('grammar_correction_info', {
                    'original_text': message,
                    'corrected_text': corrected_text,
                    'formatted_text': formatted_text,
                    'used_ai': used_ai
                }, room=request.sid)
            
        except Exception as e:
            logger.error(f"Error processing message with language processor: {e}")
            # Fallback to original message
            processed_message = message
            corrected_text = message
            formatted_text = message
    
    # Add to chat history
    if chat_manager:
        chat_manager.add_message('siva_hari_chat', user_type, processed_message)
    
    # Prepare message data
    message_data = {
        'from': user_type,
        'message': processed_message,
        'type': 'text',
        'timestamp': datetime.now().isoformat(),
        'original_text': message,
        'corrected_text': corrected_text,
        'formatted_text': formatted_text,
        'used_ai': used_ai,
        'target_language': target_language
    }
    
    # SPEECH-TO-SIGN CONNECTION - Convert Hari's message to sign sequence for Siva
    sign_sequence = []
    if user_type == 'hari' and web_converter:
        try:
            sign_sequence = web_converter.text_to_gloss_sequence(processed_message)
            message_data['sign_sequence'] = sign_sequence
            logger.info(f"🔤 Converted Hari's message to sign sequence: {sign_sequence}")
            
            # Send sign sequence to Siva for display
            siva_sid = active_users['siva']['sid']
            if siva_sid:
                # Send both the message and sign sequence
                socketio.emit('sign_sequence_update', {
                    'from': 'hari',
                    'sign_sequence': sign_sequence,
                    'original_text': processed_message,
                    'corrected_text': corrected_text,
                    'formatted_text': formatted_text,
                    'timestamp': datetime.now().isoformat()
                }, room=siva_sid)
                logger.info(f"🔄 Sent sign sequence to Siva: {sign_sequence}")
                
        except Exception as e:
            logger.error(f"❌ Error converting to sign sequence: {e}")
    
    # Broadcast message to the other user
    other_user = 'hari' if user_type == 'siva' else 'siva'
    other_sid = active_users[other_user]['sid']
    
    if other_sid:
        if user_type == 'siva':
            # Siva's message to Hari
            socketio.emit('receive_siva_message', message_data, room=other_sid)
        else:
            # Hari's message to Siva (with sign sequence)
            socketio.emit('receive_message', message_data, room=other_sid)
    
    # Send confirmation to sender with corrected text
    emit('message_sent_confirmation', {
        'message': processed_message,
        'original_text': message,
        'corrected_text': corrected_text,
        'formatted_text': formatted_text,
        'timestamp': datetime.now().isoformat(),
        'used_ai': used_ai
    })
    
    logger.info(f"💬 {user_type.capitalize()} -> {other_user.capitalize()}: {processed_message}")

@socketio.on('translate_text')
def handle_translate_text(data):
    """Handle text translation with enhanced Gemini API support"""
    text = data.get('text', '')
    target_language = data.get('target_language', 'en')
    
    if not text:
        emit('translation_error', {'error': 'No text provided for translation'})
        return
    
    try:
        if language_processor:
            logger.info(f"🌐 Translating text: '{text}' to {target_language}")
            
            if target_language == 'en':
                # For English, use grammar correction instead
                result = language_processor.process_letter_sequence(text, None)
                translated_text = result['corrected_text']
            else:
                # Use Gemini API for translation
                translated_text = language_processor.translate_text_with_gemini(text, target_language)
                
                # Fallback if translation failed
                if not translated_text or translated_text == text:
                    translated_text = language_processor.translate_text_offline(text, target_language)
            
            if translated_text and translated_text != text:
                emit('translation_result', {
                    'original_text': text,
                    'translated_text': translated_text,
                    'target_language': target_language,
                    'success': True
                })
                logger.info(f"✅ Translation successful: '{text}' → '{translated_text}'")
            else:
                emit('translation_result', {
                    'original_text': text,
                    'translated_text': text,  # Return original if translation failed
                    'target_language': target_language,
                    'success': False
                })
                logger.warning(f"⚠️ Translation returned same text for: '{text}'")
                
        else:
            # Fallback translation
            translations = {
                'hi': f'[Hindi] {text}',
                'ta': f'[Tamil] {text}',
                'en': text
            }
            
            translated_text = translations.get(target_language, text)
            
            emit('translation_result', {
                'original_text': text,
                'translated_text': translated_text,
                'target_language': target_language,
                'success': True
            })
            
    except Exception as e:
        logger.error(f"❌ Translation error: {e}")
        emit('translation_error', {'error': f'Translation failed: {str(e)}'})

@socketio.on('speak_text_direct')
def handle_speak_text_direct(data):
    """Handle direct text-to-speech with multilingual support"""
    text = data.get('text', '')
    language = data.get('language', 'en')
    
    if not text:
        emit('speech_error', {'error': 'No text provided for speech'})
        return
    
    try:
        if detector:
            # Use detector's TTS functionality
            original_lang = detector.target_language
            detector.target_language = language
            detector.speak_text(text)
            detector.target_language = original_lang
            
            emit('speech_completed', {
                'text': text,
                'language': language
            })
            logger.info(f"🔊 Spoke text: '{text}' in {language}")
        else:
            emit('speech_error', {'error': 'TTS not available'})
                
    except Exception as e:
        logger.error(f"❌ Speech error: {e}")
        emit('speech_error', {'error': f'Speech failed: {str(e)}'})

@socketio.on('get_detection_status')
def handle_get_detection_status():
    """Get current detection status"""
    if detector:
        status = {
            'active': detector.detection_active,
            'current_text': detector.get_current_text(),
            'raw_text': detector.current_letter_sequence if hasattr(detector, 'current_letter_sequence') else '',
            'processed_text': detector.processed_output if hasattr(detector, 'processed_output') else '',
            'formatted_text': detector.formatted_text if hasattr(detector, 'formatted_text') else '',
            'corrected_text': detector.corrected_text if hasattr(detector, 'corrected_text') else '',
            'hand_detected': detector.hand_in_box if hasattr(detector, 'hand_in_box') else False,
            'is_analyzing': detector.is_analyzing if hasattr(detector, 'is_analyzing') else False,
            'is_in_cooldown': detector.is_in_cooldown if hasattr(detector, 'is_in_cooldown') else False,
            'last_prediction': detector.last_prediction if hasattr(detector, 'last_prediction') else '',
            'detection_count': detector.detection_count if hasattr(detector, 'detection_count') else 0,
            'target_language': detector.target_language if hasattr(detector, 'target_language') else None
        }
        emit('detection_status', status)
    else:
        emit('detection_status', {'active': False, 'error': 'Detector not available'})

# Handle sign sequence requests
@socketio.on('request_sign_sequence')
def handle_request_sign_sequence(data):
    """Handle requests to convert text to sign sequence"""
    text = data.get('text', '')
    user_type = data.get('user_type', 'hari')
    
    if not text:
        emit('sign_sequence_error', {'error': 'No text provided'})
        return
    
    try:
        if web_converter:
            sign_sequence = web_converter.text_to_gloss_sequence(text)
            
            # Send to Siva for display
            siva_sid = active_users['siva']['sid']
            if siva_sid:
                socketio.emit('sign_sequence_update', {
                    'from': user_type,
                    'sign_sequence': sign_sequence,
                    'original_text': text,
                    'timestamp': datetime.now().isoformat()
                }, room=siva_sid)
                
                emit('sign_sequence_result', {
                    'sign_sequence': sign_sequence,
                    'original_text': text,
                    'success': True
                })
                
                logger.info(f"🔤 Generated sign sequence for '{text}': {sign_sequence}")
            else:
                emit('sign_sequence_error', {'error': 'Siva is not connected'})
        else:
            emit('sign_sequence_error', {'error': 'Sign converter not available'})
            
    except Exception as e:
        logger.error(f"❌ Sign sequence error: {e}")
        emit('sign_sequence_error', {'error': f'Failed to generate sign sequence: {str(e)}'})

# ENHANCED: Grammar correction requests
@socketio.on('correct_grammar')
def handle_correct_grammar(data):
    """Handle grammar correction requests using Gemini API"""
    text = data.get('text', '')
    
    if not text:
        emit('grammar_correction_error', {'error': 'No text provided for correction'})
        return
    
    try:
        if language_processor:
            # Use language processor for grammar correction
            logger.info(f"🔤 Correcting grammar for: '{text}'")
            result = language_processor.process_letter_sequence(text)
            
            emit('grammar_correction_result', {
                'original_text': text,
                'corrected_text': result['corrected_text'],
                'formatted_text': result['formatted_text'],
                'used_ai': result['used_ai'],
                'success': True
            })
            
            logger.info(f"🔤 Grammar correction: '{text}' -> '{result['corrected_text']}'")
        else:
            emit('grammar_correction_error', {'error': 'Language processor not available'})
            
    except Exception as e:
        logger.error(f"❌ Grammar correction error: {e}")
        emit('grammar_correction_error', {'error': f'Grammar correction failed: {str(e)}'})

# NEW: Language detection endpoint
@socketio.on('detect_language')
def handle_detect_language(data):
    """Detect language of provided text"""
    text = data.get('text', '')
    
    if not text:
        emit('language_detection_error', {'error': 'No text provided for language detection'})
        return
    
    try:
        # Simple language detection based on character ranges
        def detect_language_simple(text):
            # Check for Hindi characters (Devanagari range)
            if any('\u0900' <= char <= '\u097F' for char in text):
                return 'hi'
            # Check for Tamil characters
            elif any('\u0B80' <= char <= '\u0BFF' for char in text):
                return 'ta'
            # Default to English
            else:
                return 'en'
        
        detected_lang = detect_language_simple(text)
        
        emit('language_detection_result', {
            'text': text,
            'detected_language': detected_lang,
            'success': True
        })
        
        logger.info(f"🔤 Detected language for '{text}': {detected_lang}")
        
    except Exception as e:
        logger.error(f"❌ Language detection error: {e}")
        emit('language_detection_error', {'error': f'Language detection failed: {str(e)}'})

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Sign Language Chat Bridge Server...")
    print("=" * 60)
    print("📡 Server running on: http://localhost:5000")
    print("🌐 Frontend available at: http://localhost:5000/")
    print("👤 Siva interface: http://localhost:5000/siva")
    print("👤 Hari interface: http://localhost:5000/hari")
    print("📊 API Status: http://localhost:5000/api/status")
    print("🎥 Detection system: Starts only when 'Start Signing' is clicked")
    print("🔊 Speech and translation features enabled")
    print("🔄 Speech-to-Sign connection: ACTIVE")
    print("🔤 Gemini API Grammar Correction: ENHANCED")
    print(f"🖼️ Sign Images Loaded: {len(sign_images)}")
    print(f"🤖 Gemini AI Available: {language_processor.gemini_available if language_processor else False}")
    print("=" * 60)
    
    try:
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=True,
                    use_reloader=False,
                    allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        print(f"❌ Server failed to start: {e}")