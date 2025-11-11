/**
 * Enhanced Hari App with Translation and Speech
 * FIXED: Proper grammar correction display and sign image connection
 * FIXED: Enhanced grammar correction feedback and UI
 * FIXED: Improved multilingual translation support
 */

class HariApp {
    constructor() {
        this.socket = null;
        this.isListening = false;
        this.sessionStartTime = null;
        this.timerInterval = null;
        this.currentMessage = '';
        this.currentTranslation = '';
        this.currentOriginalText = '';
        this.messageCount = 0;
        this.recognition = null;
        this.grammarCorrections = new Map(); // Store grammar corrections
        this.isConnected = false;
    }

    init() {
        try {
            // Initialize Socket Connection
            this.initSocketConnection();
            
            // Initialize UI Components
            this.initUIComponents();
            
            // Initialize Speech Recognition
            this.initSpeechRecognition();
            
            // Start session timer
            this.startSessionTimer();
            
            console.log('✅ Hari App initialized successfully');
            
        } catch (error) {
            console.error('❌ Hari App initialization failed:', error);
            this.showError('Failed to initialize application');
        }
    }

    initSocketConnection() {
        try {
        // Use the specific IP address of your server
        this.socket = io('http://172.20.10.4:5000', {
            transports: ['websocket', 'polling'],
            timeout: 10000
        });
        
        this.socket.on('connect', () => {
            console.log('✅ Connected to server at 172.20.10.4:5000');
            this.isConnected = true;
            this.updateConnectionStatus('connected', 'Connected to server');
            
            // Join chat as Hari
            this.socket.emit('join_chat', {
                user_type: 'hari',
                room_id: 'siva_hari_chat'
            });
        });

        this.socket.on('disconnect', (reason) => {
            console.log('❌ Disconnected from server:', reason);
            this.isConnected = false;
            this.updateConnectionStatus('disconnected', `Disconnected: ${reason}`);
        });

        this.socket.on('connect_error', (error) => {
            console.error('❌ Connection error:', error);
            this.updateConnectionStatus('error', `Connection failed: ${error.message}`);
        });

            this.socket.on('join_status', (data) => {
                if (data.status === 'joined') {
                    console.log(`✅ Joined room: ${data.room_id}`);
                    this.updatePartnerStatus('connected');
                }
            });

            // Listen for detection updates from Siva
            this.socket.on('detection_update', (data) => {
                console.log('📨 Received detection update from Siva:', data);
                this.handleSivaDetection(data);
            });

            // Listen for direct messages from Siva
            this.socket.on('receive_siva_message', (data) => {
                console.log('📩 Received message from Siva:', data);
                this.handleSivaMessage(data);
            });

            this.socket.on('translation_result', (data) => {
                console.log('🌐 Translation result:', data);
                this.handleTranslationResult(data);
            });

            this.socket.on('speech_completed', (data) => {
                this.showNotification('Speech completed successfully', 'success');
            });

            this.socket.on('speech_error', (data) => {
                this.showError(`Speech error: ${data.error}`);
            });

            this.socket.on('chat_history', (history) => {
                this.loadChatHistory(history);
            });

            // Listen for user status updates
            this.socket.on('user_status', (data) => {
                if (data.user === 'siva') {
                    this.updatePartnerStatus(data.status);
                    if (data.status === 'connected') {
                        this.showNotification('Siva is now online', 'success');
                    } else {
                        this.showNotification('Siva has disconnected', 'warning');
                    }
                }
            });

            // ENHANCED: Grammar correction feedback
            this.socket.on('grammar_correction_info', (data) => {
                console.log('🔤 Grammar correction info:', data);
                this.handleGrammarCorrectionInfo(data);
            });

            this.socket.on('message_sent_confirmation', (data) => {
                console.log('✅ Message sent confirmation:', data);
                this.handleMessageConfirmation(data);
            });

            // Listen for sign sequence confirmation
            this.socket.on('sign_sequence_result', (data) => {
                if (data.success) {
                    console.log('✅ Sign sequence generated successfully:', data.sign_sequence);
                    this.showNotification('Message converted to signs for Siva', 'success');
                } else {
                    console.error('❌ Sign sequence generation failed:', data.error);
                    this.showError('Failed to convert message to signs');
                }
            });

            // Listen for grammar correction results
            this.socket.on('grammar_correction_result', (data) => {
                this.handleGrammarCorrectionResult(data);
            });

            // Listen for language detection results
            this.socket.on('language_detection_result', (data) => {
                this.handleLanguageDetectionResult(data);
            });

        } catch (error) {
            console.error('❌ Socket initialization failed:', error);
            this.updateConnectionStatus('error', 'Connection failed');
        }
    }

    initUIComponents() {
        // Send message button
        const sendBtn = document.getElementById('sendBtn');
        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }

        // Speech button
        const speechBtn = document.getElementById('speechBtn');
        if (speechBtn) {
            speechBtn.addEventListener('click', () => this.toggleSpeechRecognition());
        }

        // Quick response buttons
        const responseBtns = document.querySelectorAll('.response-btn');
        responseBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const text = e.target.dataset.text;
                this.useQuickResponse(text);
            });
        });

        // Language selection
        const languageRadios = document.querySelectorAll('input[name="language"]');
        languageRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                console.log('🌐 Language changed to:', e.target.value);
                if (this.currentMessage) {
                    this.translateMessage(this.currentMessage, e.target.value);
                } else {
                    this.displayTranslationResult('', e.target.value);
                }
            });
        });

        // Speak buttons
        const speakMessageBtn = document.getElementById('speakMessage');
        if (speakMessageBtn) {
            speakMessageBtn.addEventListener('click', () => this.speakMessage());
        }

        const speakTranslationBtn = document.getElementById('speakTranslation');
        if (speakTranslationBtn) {
            speakTranslationBtn.addEventListener('click', () => this.speakTranslation());
        }

        // Copy translation button
        const copyTranslationBtn = document.getElementById('copyTranslation');
        if (copyTranslationBtn) {
            copyTranslationBtn.addEventListener('click', () => this.copyTranslation());
        }

        // Clear chat button
        const clearChatBtn = document.getElementById('clearChat');
        if (clearChatBtn) {
            clearChatBtn.addEventListener('click', () => this.clearChat());
        }

        // Grammar correction button
        const grammarBtn = document.getElementById('correctGrammar');
        if (grammarBtn) {
            grammarBtn.addEventListener('click', () => this.correctGrammar());
        }

        // Detect language button
        const detectLangBtn = document.getElementById('detectLanguage');
        if (detectLangBtn) {
            detectLangBtn.addEventListener('click', () => this.detectLanguage());
        }

        // Text input enter key support
        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            textInput.addEventListener('input', (e) => {
                this.handleTextInput(e.target.value);
            });
        }
    }

    // ENHANCED: Grammar correction handling
    handleGrammarCorrectionInfo(data) {
        if (data.corrected_text !== data.original_text) {
            const correctionId = Date.now();
            this.grammarCorrections.set(correctionId, data);
            
            this.showGrammarCorrectionNotification(data, correctionId);
            
            // Update the last message in chat if it matches the original
            this.updateChatMessageWithCorrection(data.original_text, data.corrected_text);
        }
    }

    handleMessageConfirmation(data) {
        if (data.corrected_text && data.corrected_text !== data.original_text) {
            const correctionId = Date.now();
            this.grammarCorrections.set(correctionId, {
                original_text: data.original_text,
                corrected_text: data.corrected_text,
                formatted_text: data.formatted_text,
                used_ai: data.used_ai
            });
            
            this.showGrammarCorrectionNotification({
                original_text: data.original_text,
                corrected_text: data.corrected_text,
                formatted_text: data.formatted_text,
                used_ai: data.used_ai
            }, correctionId);
        }
    }

    showGrammarCorrectionNotification(data, correctionId) {
        const notification = document.createElement('div');
        notification.className = 'notification grammar-correction';
        notification.innerHTML = `
            <div class="grammar-correction-content">
                <div class="grammar-header">
                    <i class="fas fa-magic"></i>
                    <strong>Grammar Improved</strong>
                    <button class="close-btn" onclick="this.parentElement.parentElement.parentElement.remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="grammar-comparison">
                    <div class="original">
                        <span class="label">Original:</span>
                        <span class="text">"${this.escapeHtml(data.original_text)}"</span>
                    </div>
                    <div class="corrected">
                        <span class="label">Corrected:</span>
                        <span class="text">"${this.escapeHtml(data.corrected_text)}"</span>
                    </div>
                </div>
                <div class="grammar-footer">
                    <small>${data.used_ai ? '🤖 AI-powered correction' : '⚡ Automatic correction'}</small>
                    <button class="btn-small" onclick="window.hariApp.applyGrammarCorrection(${correctionId})">
                        Apply
                    </button>
                </div>
            </div>
        `;
        
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            width: 400px;
            background: white;
            border-left: 4px solid #10b981;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            animation: slideInRight 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 8 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOutRight 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            }
        }, 8000);
    }

    applyGrammarCorrection(correctionId) {
        const correction = this.grammarCorrections.get(correctionId);
        if (correction) {
            const textInput = document.getElementById('textInput');
            if (textInput) {
                textInput.value = correction.corrected_text;
                this.handleTextInput(correction.corrected_text);
                this.showNotification('Grammar correction applied', 'success');
            }
            this.grammarCorrections.delete(correctionId);
            
            // Remove the notification
            const notifications = document.querySelectorAll('.grammar-correction');
            notifications.forEach(notification => notification.remove());
        }
    }

    updateChatMessageWithCorrection(originalText, correctedText) {
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            const messages = chatMessages.querySelectorAll('.message.hari');
            const lastMessage = messages[messages.length - 1];
            
            if (lastMessage) {
                const messageText = lastMessage.querySelector('.message-text');
                if (messageText && messageText.textContent === originalText) {
                    messageText.textContent = correctedText;
                    
                    // Add correction indicator
                    const correctionIndicator = document.createElement('div');
                    correctionIndicator.className = 'correction-indicator';
                    correctionIndicator.innerHTML = '<i class="fas fa-magic"></i> Grammar corrected';
                    correctionIndicator.style.cssText = `
                        font-size: 0.7rem;
                        color: #10b981;
                        margin-top: 4px;
                        display: flex;
                        align-items: center;
                        gap: 4px;
                    `;
                    
                    lastMessage.querySelector('.message-content').appendChild(correctionIndicator);
                }
            }
        }
    }

    handleGrammarCorrectionResult(data) {
        if (data.success) {
            console.log('✅ Grammar correction result:', data);
            
            // Update the displayed message with corrected text if different
            if (data.corrected_text !== data.original_text) {
                this.showGrammarCorrectionNotification(data, Date.now());
                
                // Update input field with corrected text
                const textInput = document.getElementById('textInput');
                if (textInput) {
                    textInput.value = data.corrected_text;
                    this.handleTextInput(data.corrected_text);
                }
            } else {
                this.showNotification('Text is already grammatically correct', 'info');
            }
        } else {
            this.showError(`Grammar correction failed: ${data.error}`);
        }
    }

    correctGrammar() {
        const textInput = document.getElementById('textInput');
        if (!textInput || !textInput.value.trim()) {
            this.showError('Please enter some text to correct grammar');
            return;
        }

        const text = textInput.value.trim();
        this.socket.emit('correct_grammar', { text: text });
        this.showNotification('Correcting grammar with AI...', 'info');
    }

    detectLanguage() {
        const textInput = document.getElementById('textInput');
        if (!textInput || !textInput.value.trim()) {
            this.showError('Please enter some text to detect language');
            return;
        }

        const text = textInput.value.trim();
        this.socket.emit('detect_language', { text: text });
        this.showNotification('Detecting language...', 'info');
    }

    handleLanguageDetectionResult(data) {
        if (data.success) {
            this.showNotification(`Detected language: ${this.getLanguageName(data.detected_language)}`, 'success');
            
            // Auto-select the detected language in radio buttons
            const languageRadio = document.querySelector(`input[name="language"][value="${data.detected_language}"]`);
            if (languageRadio) {
                languageRadio.checked = true;
                this.showNotification(`Auto-selected ${this.getLanguageName(data.detected_language)} for translation`, 'info');
            }
        } else {
            this.showError('Language detection failed');
        }
    }

    sendMessage() {
        const textInput = document.getElementById('textInput');
        if (!textInput) return;

        const message = textInput.value.trim();
        if (!message) {
            this.showError('Please enter a message');
            return;
        }

        if (this.socket && this.isConnected) {
            const targetLanguage = document.querySelector('input[name="language"]:checked').value;
            
            // Send chat message - this will trigger grammar correction and translation on server
            this.socket.emit('send_chat_message', {
                message: message,
                user_type: 'hari',
                target_language: targetLanguage
            });

            // Also request sign sequence conversion for Siva
            this.socket.emit('request_sign_sequence', {
                text: message,
                user_type: 'hari'
            });

            this.addChatMessage('hari', message, 'text');
            textInput.value = '';
            this.handleTextInput('');
            this.showNotification('Message sent to Siva with grammar correction', 'success');
        } else {
            this.showError('Not connected to server');
        }
    }

    handleSivaDetection(data) {
        if (data.processed_text && data.processed_text.trim()) {
            console.log('🔄 Processing Siva detection:', data);
            
            this.displayReceivedMessage(
                data.processed_text, 
                data.original_text || data.raw_text, 
                new Date().toISOString(),
                data.corrected_text
            );
            
            const selectedLanguage = document.querySelector('input[name="language"]:checked').value;
            
            if (selectedLanguage !== 'en' && data.processed_text) {
                this.translateMessage(data.processed_text, selectedLanguage);
            } else {
                this.displayTranslationResult(data.processed_text, 'en');
            }
            
            this.playNotificationSound();
            this.addChatMessage('siva', data.processed_text, 'sign_text', data.corrected_text);
            
            // Show grammar correction info if available
            if (data.corrected_text && data.corrected_text !== data.processed_text) {
                this.showNotification(`Siva's message was grammar corrected`, 'info');
            }
        }
    }

    handleSivaMessage(data) {
        if (data.from === 'siva' && data.message && data.message.trim()) {
            console.log('📝 Processing Siva message:', data);
            
            this.displayReceivedMessage(
                data.message, 
                data.original_text || data.message, 
                data.timestamp || new Date().toISOString(),
                data.corrected_text
            );
            
            const selectedLanguage = document.querySelector('input[name="language"]:checked').value;
            
            if (selectedLanguage !== 'en') {
                this.translateMessage(data.message, selectedLanguage);
            } else {
                this.displayTranslationResult(data.message, 'en');
            }
            
            this.playNotificationSound();
            this.addChatMessage('siva', data.message, 'text', data.corrected_text);
            
            // Show grammar correction info if available
            if (data.corrected_text && data.corrected_text !== data.message) {
                this.showNotification(`Siva's message was grammar corrected`, 'info');
            }
        }
    }

    displayReceivedMessage(message, originalText, timestamp, correctedText = '') {
        console.log('📝 Displaying message:', message);
        
        this.currentMessage = message;
        this.currentOriginalText = originalText || message;
        
        const receivedMessage = document.getElementById('receivedMessage');
        const messageTime = document.getElementById('messageTime');
        
        if (receivedMessage) {
            if (message && message.trim()) {
                let correctionHtml = '';
                if (correctedText && correctedText !== originalText) {
                    correctionHtml = `
                        <div class="correction-info">
                            <i class="fas fa-magic"></i>
                            <span>Grammar corrected from "${this.escapeHtml(originalText)}"</span>
                        </div>
                    `;
                }
                
                receivedMessage.innerHTML = `
                    <div class="message-content">
                        <div class="message-header">
                            <strong>From Siva</strong>
                            <span class="message-time">${new Date(timestamp).toLocaleTimeString()}</span>
                        </div>
                        <div class="message-text">${this.escapeHtml(message)}</div>
                        ${originalText && originalText !== message ? 
                         `<div class="original-text">
                             <strong>Original signs:</strong> ${this.escapeHtml(originalText)}
                         </div>` : ''}
                         ${correctionHtml}
                    </div>
                `;
            } else {
                receivedMessage.innerHTML = '<div class="placeholder">Waiting for message from Siva...</div>';
            }
        }
        
        if (messageTime) {
            const time = new Date(timestamp).toLocaleTimeString();
            messageTime.textContent = `Received: ${time}`;
        }
        
        this.messageCount++;
        this.updateMessageCount();
    }

    translateMessage(text, targetLanguage) {
        if (!text || !text.trim()) {
            console.log('⚠️ No text to translate');
            return;
        }
        
        if (!this.socket || !this.isConnected) {
            this.showError('Not connected to server');
            return;
        }
        
        try {
            console.log(`🌐 Translating: "${text}" to ${targetLanguage}`);
            this.showNotification('Translating with AI...', 'info');
            
            this.socket.emit('translate_text', {
                text: text,
                target_language: targetLanguage
            });
            
        } catch (error) {
            console.error('Translation error:', error);
            this.showError('Translation failed');
        }
    }

    handleTranslationResult(data) {
        console.log('🔄 Handling translation result:', data);
        
        this.currentTranslation = data.translated_text;
        
        const translatedMessage = document.getElementById('translatedMessage');
        if (translatedMessage) {
            if (data.translated_text && data.translated_text.trim()) {
                const successBadge = data.success ? '<span class="success-badge">✓ AI Translation</span>' : '<span class="warning-badge">⚠ Basic Translation</span>';
                
                translatedMessage.innerHTML = `
                    <div class="message-content">
                        <div class="message-header">
                            <strong>Translation</strong>
                            <span class="message-time">${this.getLanguageName(data.target_language)} ${successBadge}</span>
                        </div>
                        <div class="message-text">${this.escapeHtml(data.translated_text)}</div>
                        <div class="translation-info">
                            <i class="fas fa-language"></i> Translated from English to ${this.getLanguageName(data.target_language)}
                        </div>
                    </div>
                `;
            } else {
                translatedMessage.innerHTML = '<div class="placeholder">No translation available</div>';
            }
        }
        
        const messageLanguage = document.getElementById('messageLanguage');
        if (messageLanguage) {
            messageLanguage.textContent = this.getLanguageName(data.target_language);
        }
        
        if (data.success) {
            this.showNotification('AI translation completed', 'success');
        } else {
            this.showNotification('Used basic translation', 'warning');
        }
    }

    displayTranslationResult(text, language) {
        this.currentTranslation = text;
        
        const translatedMessage = document.getElementById('translatedMessage');
        if (translatedMessage) {
            if (text && text.trim()) {
                translatedMessage.innerHTML = `
                    <div class="message-content">
                        <div class="message-header">
                            <strong>Translation</strong>
                            <span class="message-time">${this.getLanguageName(language)}</span>
                        </div>
                        <div class="message-text">${this.escapeHtml(text)}</div>
                    </div>
                `;
            } else {
                translatedMessage.innerHTML = '<div class="placeholder">Select language to see translation...</div>';
            }
        }
        
        const messageLanguage = document.getElementById('messageLanguage');
        if (messageLanguage && text && text.trim()) {
            messageLanguage.textContent = this.getLanguageName(language);
        }
    }

    addChatMessage(sender, message, type = 'text', correctedText = '') {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        let correctionHtml = '';
        if (correctedText && correctedText !== message) {
            correctionHtml = `
                <div class="correction-indicator">
                    <i class="fas fa-magic"></i> Grammar corrected
                </div>
            `;
        }
        
        messageContent.innerHTML = `
            <div class="message-header">
                <strong>${sender === 'siva' ? 'Siva' : 'You'}</strong>
                <span class="message-time">${this.getCurrentTime()}</span>
            </div>
            <div class="message-text">${this.escapeHtml(message)}</div>
            ${correctionHtml}
        `;
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        this.messageCount++;
        this.updateMessageCount();
    }

    loadChatHistory(history) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        const systemMessages = chatMessages.querySelectorAll('.system-message');
        chatMessages.innerHTML = '';
        systemMessages.forEach(msg => chatMessages.appendChild(msg));

        history.forEach(msg => {
            if (msg.from !== 'system') {
                this.addChatMessage(msg.from, msg.message, msg.type);
            }
        });
        
        this.messageCount = history.filter(msg => msg.from !== 'system').length;
        this.updateMessageCount();
    }

    useQuickResponse(text) {
        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.value = text;
            this.handleTextInput(text);
            this.showNotification('Quick response added to input', 'info');
        }
    }

    speakMessage() {
        if (!this.currentMessage || !this.currentMessage.trim()) {
            this.showError('No message to speak');
            return;
        }
        
        console.log('🔊 Speaking message:', this.currentMessage);
        this.speakText(this.currentMessage, 'en');
    }

    speakTranslation() {
        if (!this.currentTranslation || !this.currentTranslation.trim()) {
            this.showError('No translation to speak');
            return;
        }
        
        const selectedLanguage = document.querySelector('input[name="language"]:checked').value;
        console.log('🔊 Speaking translation:', this.currentTranslation, 'in', selectedLanguage);
        this.speakText(this.currentTranslation, selectedLanguage);
    }

    speakText(text, language) {
        if (!this.socket || !this.isConnected) {
            this.showError('Not connected to server');
            return;
        }
        
        if (!text || !text.trim()) {
            this.showError('No text to speak');
            return;
        }
        
        this.socket.emit('speak_text_direct', {
            text: text,
            language: language
        });
        
        this.showNotification(`Speaking in ${this.getLanguageName(language)}...`, 'info');
    }

    copyTranslation() {
        if (!this.currentTranslation || !this.currentTranslation.trim()) {
            this.showError('No translation to copy');
            return;
        }
        
        navigator.clipboard.writeText(this.currentTranslation).then(() => {
            this.showNotification('Translation copied to clipboard', 'success');
        }).catch(() => {
            this.showError('Failed to copy translation');
        });
    }

    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            const systemMessages = chatMessages.querySelectorAll('.system-message');
            chatMessages.innerHTML = '';
            systemMessages.forEach(msg => chatMessages.appendChild(msg));
        }
        
        this.currentMessage = '';
        this.currentTranslation = '';
        const receivedMessage = document.getElementById('receivedMessage');
        const translatedMessage = document.getElementById('translatedMessage');
        
        if (receivedMessage) {
            receivedMessage.innerHTML = '<div class="placeholder">Waiting for message from Siva...</div>';
        }
        if (translatedMessage) {
            translatedMessage.innerHTML = '<div class="placeholder">Select language to see translation...</div>';
        }
        
        const englishRadio = document.querySelector('input[name="language"][value="en"]');
        if (englishRadio) {
            englishRadio.checked = true;
        }
        
        const messageLanguage = document.getElementById('messageLanguage');
        if (messageLanguage) {
            messageLanguage.textContent = 'English';
        }
        
        this.messageCount = 0;
        this.updateMessageCount();
        this.showNotification('Chat cleared', 'success');
    }

    handleTextInput(text) {
        const sendBtn = document.getElementById('sendBtn');
        if (sendBtn) {
            sendBtn.disabled = !text.trim();
        }
        this.updateTextStats(text);
    }

    updateTextStats(text) {
        const charCount = text.length;
        const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
        
        document.getElementById('characterCount').textContent = `${charCount} characters`;
        document.getElementById('wordCount').textContent = `${wordCount} words`;
    }

    updateMessageCount() {
        const messageCountElement = document.getElementById('messageCount');
        if (messageCountElement) {
            messageCountElement.textContent = `${this.messageCount} messages`;
        }
        
        const sessionTimer = document.getElementById('sessionTimer');
        if (sessionTimer && this.messageCount > 0) {
            sessionTimer.textContent = `${this.messageCount} msgs`;
        }
    }

    initSpeechRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn('Speech recognition not supported');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
            this.isListening = true;
            this.updateSpeechButton(true);
            this.showNotification('Listening... Speak now', 'info');
        };

        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            const textInput = document.getElementById('textInput');
            if (textInput) {
                textInput.value = transcript;
                this.handleTextInput(transcript);
                this.showNotification('Speech recognized successfully', 'success');
            }
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.showError(`Speech recognition error: ${event.error}`);
        };

        this.recognition.onend = () => {
            this.isListening = false;
            this.updateSpeechButton(false);
        };
    }

    toggleSpeechRecognition() {
        if (!this.recognition) {
            this.showError('Speech recognition not supported');
            return;
        }

        if (this.isListening) {
            this.recognition.stop();
        } else {
            try {
                this.recognition.start();
            } catch (error) {
                this.showError('Speech recognition failed to start');
            }
        }
    }

    updateSpeechButton(listening) {
        const speechBtn = document.getElementById('speechBtn');
        if (speechBtn) {
            if (listening) {
                speechBtn.classList.add('listening');
                speechBtn.innerHTML = '<i class="fas fa-stop"></i> Stop';
            } else {
                speechBtn.classList.remove('listening');
                speechBtn.innerHTML = '<i class="fas fa-microphone"></i> Voice';
            }
        }
    }

    startSessionTimer() {
        this.sessionStartTime = new Date();
        this.timerInterval = setInterval(() => {
            this.updateSessionTimer();
        }, 1000);
    }

    updateSessionTimer() {
        const timerElement = document.getElementById('sessionTimer');
        if (!timerElement) return;

        const now = new Date();
        const diff = Math.floor((now - this.sessionStartTime) / 1000);
        const minutes = Math.floor(diff / 60);
        const seconds = diff % 60;

        timerElement.textContent = 
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    updateConnectionStatus(status, message) {
        const statusElement = document.getElementById('connectionStatus');
        const networkStatus = document.getElementById('networkStatus');
        
        if (statusElement) {
            statusElement.innerHTML = `<i class="fas fa-circle"></i><span>${message}</span>`;
            if (status === 'connected') {
                statusElement.classList.add('connected');
            } else {
                statusElement.classList.remove('connected');
            }
        }
        
        if (networkStatus) {
            networkStatus.textContent = message;
        }
    }

    updatePartnerStatus(status) {
        const partnerStatus = document.getElementById('partnerStatus');
        if (partnerStatus) {
            partnerStatus.textContent = status === 'connected' ? 
                'Siva is online' : 'Waiting for Siva...';
        }
    }

    playNotificationSound() {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = 800;
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0, audioContext.currentTime);
            gainNode.gain.linearRampToValueAtTime(0.1, audioContext.currentTime + 0.01);
            gainNode.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.2);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.2);
            
        } catch (error) {
            console.log('Notification sound not supported');
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${this.getNotificationIcon(type)}"></i>
            <span>${message}</span>
        `;
        
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            background: ${this.getNotificationColor(type)};
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            animation: slideInRight 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    getNotificationIcon(type) {
        const icons = {
            'success': 'check-circle',
            'error': 'exclamation-circle',
            'info': 'info-circle',
            'warning': 'exclamation-triangle'
        };
        return icons[type] || 'info-circle';
    }

    getNotificationColor(type) {
        const colors = {
            'success': '#10b981',
            'error': '#ef4444',
            'info': '#3b82f6',
            'warning': '#f59e0b'
        };
        return colors[type] || '#3b82f6';
    }

    getLanguageName(code) {
        const languages = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil'
        };
        return languages[code] || code;
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    getCurrentTime() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Cleanup method
    destroy() {
        if (this.recognition) {
            this.recognition.stop();
        }
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }
        if (this.socket) {
            this.socket.disconnect();
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    const hariApp = new HariApp();
    hariApp.init();
    window.hariApp = hariApp;
});