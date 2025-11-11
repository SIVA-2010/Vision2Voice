"""
ENHANCED LANGUAGE PROCESSOR WITH CORRECT GEMINI MODELS
FIXED: Uses models/gemini-pro-latest for translation and grammar correction
"""

import re
import nltk
import os
import time
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class LanguageProcessor:
    def __init__(self):
        print("🔤 Initializing Enhanced Language Processor with Gemini AI...")
        self.device = "cpu"
        
        # Gemini API Configuration
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_available = False
        self.model = None
        self.available_models = []
        
        if not self.api_key:
            print("❌ GEMINI_API_KEY not found in .env file")
            print("💡 Please add your API key to .env file: GEMINI_API_KEY=your_api_key_here")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.discover_available_models()
                self.initialize_correct_model()
            except Exception as e:
                print(f"❌ Gemini AI initialization failed: {e}")
        
        # Enhanced translation cache
        self.translation_cache = {}
        
    def discover_available_models(self):
        """Discover available Gemini models"""
        try:
            print("🔍 Discovering available Gemini models...")
            models = genai.list_models()
            
            # Convert generator to list
            models_list = list(models)
            print(f"📋 Found {len(models_list)} total models")
            
            # Filter for models that support generateContent
            available_models = []
            for model in models_list:
                if 'generateContent' in model.supported_generation_methods:
                    available_models.append(model.name)
                    print(f"   ✅ Available: {model.name}")
            
            self.available_models = available_models
            print(f"🎯 {len(available_models)} models support generateContent")
            
            if not available_models:
                print("⚠️ No models found supporting generateContent!")
                
        except Exception as e:
            print(f"❌ Failed to discover models: {e}")
            # Fallback to known models
            self.available_models = [
                'models/gemini-pro-latest',
                'models/gemini-pro',
                'models/gemini-1.0-pro'
            ]
    
    def initialize_correct_model(self):
        """Initialize the correct Gemini model"""
        # Preferred model order based on your available models
        preferred_models = [
            'models/gemini-pro-latest',  # Your available model
            'models/gemini-1.5-pro',
            'models/gemini-1.5-flash', 
            'models/gemini-pro',
            'models/gemini-1.0-pro'
        ]
        
        # Filter to only available models
        available_preferred = [model for model in preferred_models if model in self.available_models]
        
        if not available_preferred and self.available_models:
            # Use first available model
            model_name = self.available_models[0]
        elif available_preferred:
            model_name = available_preferred[0]
        else:
            # Fallback
            model_name = 'models/gemini-pro-latest'
        
        try:
            print(f"🚀 Attempting to initialize model: {model_name}")
            self.model = genai.GenerativeModel(model_name)
            
            # Test the model with a simple prompt
            test_response = self.model.generate_content("Hello, translate 'good morning' to Hindi")
            if test_response and test_response.text:
                self.gemini_available = True
                print(f"✅ Successfully initialized: {model_name}")
                print(f"🎯 Model test response: {test_response.text}")
            else:
                print(f"❌ Model test failed: {model_name}")
                self.gemini_available = False
                
        except Exception as e:
            print(f"❌ Failed to initialize {model_name}: {e}")
            self.try_fallback_models(preferred_models, model_name)
    
    def try_fallback_models(self, preferred_models, failed_model):
        """Try fallback models if primary model fails"""
        # Remove the failed model from the list
        fallback_models = [model for model in preferred_models if model != failed_model]
        
        for model_name in fallback_models:
            # Only try models that are available
            if model_name not in self.available_models:
                continue
                
            try:
                print(f"🔄 Trying fallback model: {model_name}")
                self.model = genai.GenerativeModel(model_name)
                
                # Test the model with translation
                test_response = self.model.generate_content("Translate 'hello' to Hindi")
                if test_response and test_response.text:
                    self.gemini_available = True
                    print(f"✅ Successfully initialized fallback: {model_name}")
                    print(f"🎯 Fallback test response: {test_response.text}")
                    return
                    
            except Exception as e:
                print(f"❌ Fallback model {model_name} failed: {e}")
                continue
        
        # If all models fail
        print("❌ All Gemini models failed. Switching to offline mode.")
        self.gemini_available = False
        
    def query_gemini_api(self, prompt, max_retries=3):
        """Enhanced Gemini API query with better error handling"""
        if not self.gemini_available or not self.model:
            logger.info("❌ Gemini not available, using offline mode")
            return None
            
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Querying Gemini API (attempt {attempt + 1})...")
                
                generation_config = {
                    "temperature": 0.1,  # Low temperature for consistent translations
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
                
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    },
                ]
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                if response and response.text:
                    cleaned_text = response.text.strip()
                    logger.info(f"✅ Gemini response received: {cleaned_text}")
                    return cleaned_text
                else:
                    logger.warning(f"⚠️ Gemini returned empty response (attempt {attempt + 1})")
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"⚠️ Gemini API error (attempt {attempt + 1}): {error_msg}")
                
                if "quota" in error_msg.lower():
                    logger.error("❌ API quota exceeded. Switching to offline mode.")
                    self.gemini_available = False
                    break
                elif "429" in error_msg:
                    logger.warning("⏳ Rate limited, waiting before retry...")
                    time.sleep(5)
                elif "404" in error_msg:
                    logger.error("❌ Model not found. Switching to offline mode.")
                    self.gemini_available = False
                    break
                elif attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                    
        return None
    
    def correct_grammar_with_gemini(self, text):
        """Enhanced grammar correction with Gemini AI"""
        if not text or len(text.strip()) < 2:
            return text
            
        logger.info(f"🤖 Correcting grammar with Gemini: '{text}'")
        
        prompt = f"""Please correct the grammar, spelling, and punctuation of this English text. 
        Return ONLY the corrected version without any explanations, notes, or additional text.
        Preserve the original meaning and intent.
        If the text is already correct, return it as is.

        Text to correct: "{text}"

        Corrected version:"""
        
        corrected_text = self.query_gemini_api(prompt)
        
        if corrected_text:
            # Enhanced cleaning of response
            corrected_text = self.clean_gemini_response(corrected_text)
            
            if corrected_text and corrected_text != text:
                logger.info(f"✅ Grammar corrected: '{text}' → '{corrected_text}'")
                return corrected_text
        
        logger.info("❌ Gemini correction failed, using enhanced offline method")
        return self.correct_grammar_offline(text)
    
    def translate_text_with_gemini(self, text, target_lang):
        """Enhanced translation with Gemini AI for Hindi and Tamil"""
        if not text or not target_lang or target_lang == 'en':
            return text
            
        # Check cache first
        cache_key = f"{text}_{target_lang}"
        if cache_key in self.translation_cache:
            logger.info(f"📦 Using cached translation for: '{text}'")
            return self.translation_cache[cache_key]
            
        logger.info(f"🤖 Translating to {target_lang} with Gemini: '{text}'")
        
        language_names = {
            'hi': 'Hindi',
            'ta': 'Tamil',
            'en': 'English'
        }
        
        target_lang_name = language_names.get(target_lang, target_lang)
        
        prompt = f"""Translate the following English text to {target_lang_name}. 
        Return ONLY the translation without any explanations, notes, or additional text.
        Preserve the meaning, tone, and context accurately.
        Provide natural, fluent translation that sounds native.

        English text: "{text}"

        {target_lang_name} translation:"""
        
        translated_text = self.query_gemini_api(prompt)
        
        if translated_text:
            translated_text = self.clean_gemini_response(translated_text)
            
            if translated_text and translated_text != text:
                # Cache the translation
                self.translation_cache[cache_key] = translated_text
                logger.info(f"✅ Translated: '{text}' → '{translated_text}'")
                return translated_text
        
        logger.info(f"❌ Gemini translation failed, using offline method for {target_lang}")
        offline_translation = self.translate_text_offline(text, target_lang)
        self.translation_cache[cache_key] = offline_translation
        return offline_translation
    
    def clean_gemini_response(self, text):
        """Clean and normalize Gemini API responses"""
        if not text:
            return text
            
        text = text.strip()
        
        # Remove quotes if present
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        elif text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        # Remove common prefixes
        prefixes = [
            'corrected:', 'fixed:', 'version:', 'here is the corrected text:', 
            'the corrected text is:', 'corrected text:', 'grammar corrected:',
            'improved version:', 'better version:', 'translation:', 'translated:',
            'here is the translation:', 'the translation is:'
        ]
        
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove markdown formatting
        text = re.sub(r'[*_`#]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper capitalization for sentences
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        
        return text
    
    # ========== ENHANCED OFFLINE PROCESSING ==========
    
    def preprocess_letter_sequence(self, letter_sequence: str) -> str:
        """Enhanced preprocessing of letter sequences"""
        if not letter_sequence:
            return ""
        
        # Replace special tokens with spaces and punctuation
        processed = (letter_sequence
                   .replace('SPACE', ' ')
                   .replace('ENTER', '. ')
                   .replace('BACKSPACE', ''))
        
        # Remove extra spaces and normalize
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def _smart_word_formation(self, text: str) -> str:
        """Smart word formation from individual letters"""
        if not text:
            return ""
            
        words = text.split()
        formed_words = []
        current_word = ""
        
        for item in words:
            if len(item) == 1 and item.isalpha():
                current_word += item
            else:
                if current_word:
                    formed_words.append(current_word)
                    current_word = ""
                formed_words.append(item)
        
        if current_word:
            formed_words.append(current_word)
        
        return ' '.join(formed_words)
    
    def _apply_enhanced_grammar_rules(self, text: str) -> str:
        """Apply enhanced grammar rules"""
        if not text:
            return text
        
        # Capitalize first letter
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        
        # Fix common spacing issues with punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])(?=[A-Za-z])', r'\1 ', text)
        
        # Fix common contractions
        corrections = {
            r'\bi\s+m\b': "I'm",
            r'\bim\b': "I'm",
            r'\bid\b': "I'd", 
            r'\bive\b': "I've",
            r'\bill\b': "I'll",
            r'\byoure\b': "you're",
            r'\byour\s+': "you're ",
            r'\bdont\b': "don't",
            r'\bdoesnt\b': "doesn't",
            r'\bdidnt\b': "didn't",
            r'\bwont\b': "won't",
            r'\bcant\b': "can't",
            r'\bcouldnt\b': "couldn't",
            r'\bshouldnt\b': "shouldn't",
            r'\btheyre\b': "they're",
            r'\bwere\b': "we're",
        }
        
        for wrong, correct in corrections.items():
            text = re.sub(wrong, correct, text, flags=re.IGNORECASE)
        
        # Fix common spelling mistakes
        spelling_corrections = {
            r'\bteh\b': "the",
            r'\badn\b': "and", 
            r'\bthier\b': "their",
            r'\brecieve\b': "receive",
            r'\bseperate\b': "separate",
            r'\bdefinately\b': "definitely",
        }
        
        for wrong, correct in spelling_corrections.items():
            text = re.sub(wrong, correct, text, flags=re.IGNORECASE)
        
        # Ensure proper sentence endings
        if text and text[-1] not in ['.', '!', '?']:
            if len(text.split()) > 3:
                text += '.'
        
        return text
    
    def correct_grammar_offline(self, text: str) -> str:
        """Enhanced offline grammar correction"""
        if not text or len(text.strip()) < 2:
            return text
        
        # Step 1: Smart word formation
        text = self._smart_word_formation(text)
        
        # Step 2: Apply enhanced grammar rules
        text = self._apply_enhanced_grammar_rules(text)
        
        return text
    
    def translate_text_offline(self, text: str, target_lang: str) -> str:
        """Enhanced offline translation with better Hindi and Tamil support"""
        if not text or not target_lang or target_lang == 'en':
            return text
        
        # Enhanced translation dictionary
        translation_dict = {
            'hi': {
                'hello': 'नमस्ते',
                'hi': 'नमस्ते', 
                'thank you': 'धन्यवाद',
                'thanks': 'धन्यवाद',
                'please': 'कृपया',
                'sorry': 'क्षमा करें',
                'yes': 'हाँ',
                'no': 'नहीं',
                'water': 'पानी',
                'food': 'भोजन',
                'help': 'मदद',
                'home': 'घर',
                'school': 'स्कूल',
                'work': 'काम',
                'family': 'परिवार',
                'friend': 'दोस्त',
                'love': 'प्यार',
                'good': 'अच्छा',
                'bad': 'बुरा',
                'morning': 'सुबह',
                'night': 'रात',
                'how are you': 'आप कैसे हैं',
                'what is your name': 'आपका नाम क्या है',
                'my name is': 'मेरा नाम है',
                'i love you': 'मैं तुमसे प्यार करता हूँ',
                'good morning': 'शुभ प्रभात',
                'good night': 'शुभ रात्रि',
                'hello world': 'नमस्ते दुनिया',
                'thank you very much': 'बहुत बहुत धन्यवाद',
                'how is the weather': 'मौसम कैसा है',
                'where are you': 'आप कहाँ हैं',
                'what time is it': 'क्या समय हुआ है',
                'i': 'मैं',
                'you': 'आप',
                'he': 'वह',
                'she': 'वह',
                'we': 'हम',
                'they': 'वे',
                'this': 'यह',
                'that': 'वह',
                'here': 'यहाँ',
                'there': 'वहाँ'
            },
            'ta': {
                'hello': 'வணக்கம்',
                'hi': 'வணக்கம்',
                'thank you': 'நன்றி',
                'thanks': 'நன்றி',
                'please': 'தயவு செய்து',
                'sorry': 'மன்னிக்கவும்',
                'yes': 'ஆம்',
                'no': 'இல்லை',
                'water': 'தண்ணீர்',
                'food': 'உணவு',
                'help': 'உதவி',
                'home': 'வீடு',
                'school': 'பள்ளி',
                'work': 'வேலை',
                'family': 'குடும்பம்',
                'friend': 'நண்பர்',
                'love': 'காதல்',
                'good': 'நல்ல',
                'bad': 'கெட்ட',
                'morning': 'காலை',
                'night': 'இரவு',
                'how are you': 'நீங்கள் எப்படி இருக்கிறீர்கள்',
                'what is your name': 'உங்கள் பெயர் என்ன',
                'my name is': 'என் பெயர்',
                'i love you': 'நான் உன்னை காதலிக்கிறேன்',
                'good morning': 'காலை வணக்கம்',
                'good night': 'இரவு வணக்கம்',
                'hello world': 'வணக்கம் உலகம்',
                'thank you very much': 'மிக்க நன்றி',
                'how is the weather': 'வானிலை எப்படி இருக்கிறது',
                'where are you': 'நீங்கள் எங்கே இருக்கிறீர்கள்',
                'what time is it': 'என்ன நேரம் ஆகிறது',
                'i': 'நான்',
                'you': 'நீங்கள்',
                'he': 'அவன்',
                'she': 'அவள்',
                'we': 'நாங்கள்',
                'they': 'அவர்கள்',
                'this': 'இது',
                'that': 'அது',
                'here': 'இங்கே',
                'there': 'அங்கே'
            }
        }
        
        if target_lang not in translation_dict:
            return text
        
        original_lower = text.lower().strip()
        translated_text = original_lower
        lang_dict = translation_dict[target_lang]
        
        # Replace phrases (longer first to avoid partial matches)
        for english, translation in sorted(lang_dict.items(), key=lambda x: len(x[0]), reverse=True):
            if english in translated_text:
                translated_text = translated_text.replace(english, translation)
        
        # If translation occurred and it's different from original
        if translated_text != original_lower:
            # Capitalize first letter
            if translated_text:
                translated_text = translated_text[0].upper() + translated_text[1:]
            
            logger.info(f"🌐 Translated offline '{text}' to {target_lang}: '{translated_text}'")
            return translated_text
        
        return text
    
    def process_letter_sequence(self, letter_sequence: str, target_language: str = None) -> dict:
        """Main processing function with enhanced Gemini AI integration"""
        logger.info(f"🔤 Processing: '{letter_sequence}'")
        
        # Step 1: Basic formatting
        raw_letters = letter_sequence
        formatted_text = self.preprocess_letter_sequence(letter_sequence)
        logger.info(f"   📝 Formatted: '{formatted_text}'")
        
        # Step 2: Enhanced grammar correction (Gemini first, then enhanced offline)
        if self.gemini_available:
            corrected_text = self.correct_grammar_with_gemini(formatted_text)
        else:
            corrected_text = self.correct_grammar_offline(formatted_text)
        logger.info(f"   ✨ Corrected: '{corrected_text}'")
        
        # Step 3: Translation if target language specified
        translated_text = corrected_text
        if target_language and target_language != 'en':
            if self.gemini_available:
                translated_text = self.translate_text_with_gemini(corrected_text, target_language)
            else:
                translated_text = self.translate_text_offline(corrected_text, target_language)
            logger.info(f"   🌐 Translated: '{translated_text}'")
        
        return {
            'raw_letters': raw_letters,
            'formatted_text': formatted_text,
            'corrected_text': corrected_text,
            'translated_text': translated_text,
            'target_language': target_language,
            'used_ai': self.gemini_available,
            'model_available': self.gemini_available
        }
    
    def get_status(self):
        """Get current status of the language processor"""
        return {
            'gemini_available': self.gemini_available,
            'available_models': self.available_models,
            'current_model': self.model._model_name if self.model else None,
            'translation_cache_size': len(self.translation_cache)
        }

# Global instance
language_processor = LanguageProcessor()