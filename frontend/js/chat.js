/**
 * Chat management and Socket.IO communication for both Siva and Hari
 */

class ChatManager {
    constructor(userType) {
        this.userType = userType; // 'siva' or 'hari'
        this.socket = null;
        this.roomId = 'siva_hari_chat';
        this.isConnected = false;
    }

    init() {
        this.initSocketConnection();
        this.setupEventListeners();
    }

    initSocketConnection() {
        try {
            this.socket = io('http://localhost:5000');
            
            this.socket.on('connect', () => {
                console.log(`✅ ${this.userType} connected to server`);
                this.isConnected = true;
                this.updateConnectionStatus('connected', 'Connected to server');
                
                // Join chat room
                this.socket.emit('join_chat', {
                    user_type: this.userType,
                    room_id: this.roomId
                });
            });

            this.socket.on('disconnect', () => {
                console.log(`❌ ${this.userType} disconnected from server`);
                this.isConnected = false;
                this.updateConnectionStatus('disconnected', 'Disconnected from server');
            });

            this.socket.on('connect_error', (error) => {
                console.error(`❌ ${this.userType} connection error:`, error);
                this.updateConnectionStatus('error', 'Connection failed');
            });

            this.socket.on('join_status', (data) => {
                if (data.status === 'joined') {
                    console.log(`✅ Joined room: ${data.room_id}`);
                    this.updatePartnerStatus('connected');
                }
            });

            this.socket.on('chat_history', (history) => {
                this.loadChatHistory(history);
            });

            this.socket.on('receive_message', (data) => {
                this.handleIncomingMessage(data);
            });

            this.socket.on('sign_sequence_result', (data) => {
                this.handleSignSequenceResult(data);
            });

            this.socket.on('speech_result', (data) => {
                this.handleSpeechResult(data);
            });

        } catch (error) {
            console.error(`❌ ${this.userType} socket initialization failed:`, error);
            this.updateConnectionStatus('error', 'Connection failed');
        }
    }

    setupEventListeners() {
        // Common event listeners for both users
        document.addEventListener('DOMContentLoaded', () => {
            // Clear chat button
            const clearChatBtn = document.getElementById('clearChat');
            if (clearChatBtn) {
                clearChatBtn.addEventListener('click', () => this.clearChat());
            }

            // Language selection
            const languageSelect = document.getElementById('languageSelect');
            if (languageSelect) {
                languageSelect.addEventListener('change', (e) => this.changeLanguage(e.target.value));
            }

            // Text input handling
            const textInput = document.getElementById('textInput');
            if (textInput) {
                textInput.addEventListener('input', (e) => this.handleTextInput(e.target.value));
            }
        });
    }

    sendMessage(message, messageType = 'text') {
        if (!this.isConnected || !this.socket) {
            this.showError('Not connected to server');
            return false;
        }

        if (!message || !message.trim()) {
            this.showError('Message cannot be empty');
            return false;
        }

        try {
            this.socket.emit('send_message', {
                message: message.trim(),
                type: messageType,
                timestamp: Date.now()
            });

            // Add to local chat immediately
            this.addChatMessage(this.userType, message, messageType);
            
            this.showNotification('Message sent', 'success');
            return true;
            
        } catch (error) {
            console.error('❌ Error sending message:', error);
            this.showError('Failed to send message');
            return false;
        }
    }

    handleIncomingMessage(data) {
        if (data.from !== this.userType) { // Only show messages from other user
            this.addChatMessage(data.from, data.message, data.type || 'text');
            
            // If this is Siva and we received a sign sequence, show it
            if (this.userType === 'siva' && data.sign_sequence && window.signDisplay) {
                window.signDisplay.showSignSequence(data.sign_sequence, data.message);
            }
            
            // Play notification sound
            this.playNotificationSound();
        }
    }

    addChatMessage(sender, message, type = 'text') {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Format message based on type
        if (type === 'sign_text') {
            messageContent.innerHTML = `
                <div class="message-header">
                    <strong>${sender === 'siva' ? 'Siva' : 'Hari'}</strong>
                    <span class="message-time">${this.getCurrentTime()}</span>
                </div>
                <div class="message-text">${this.formatSignText(message)}</div>
            `;
        } else {
            messageContent.innerHTML = `
                <div class="message-header">
                    <strong>${sender === 'siva' ? 'Siva' : 'Hari'}</strong>
                    <span class="message-time">${this.getCurrentTime()}</span>
                </div>
                <div class="message-text">${this.escapeHtml(message)}</div>
            `;
        }
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Update character counts
        this.updateTextStats(message);
    }

    formatSignText(text) {
        return text
            .replace(/SPACE/g, '<span class="special-char">␣</span>')
            .replace(/ENTER/g, '<span class="special-char">↵</span>')
            .replace(/BACKSPACE/g, '<span class="special-char">⌫</span>');
    }

    loadChatHistory(history) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        // Clear existing messages (except system messages)
        const systemMessages = chatMessages.querySelectorAll('.system-message');
        chatMessages.innerHTML = '';
        systemMessages.forEach(msg => chatMessages.appendChild(msg));

        // Add history messages
        history.forEach(msg => {
            if (msg.from !== 'system') {
                this.addChatMessage(msg.from, msg.message, msg.type);
            }
        });
    }

    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        // Keep only system messages
        const systemMessages = chatMessages.querySelectorAll('.system-message');
        chatMessages.innerHTML = '';
        systemMessages.forEach(msg => chatMessages.appendChild(msg));
        
        this.showNotification('Chat cleared', 'success');
    }

    handleTextInput(text) {
        // Update preview if available
        const previewText = document.getElementById('previewText');
        if (previewText) {
            if (text.trim()) {
                previewText.textContent = text;
                previewText.classList.remove('placeholder');
            } else {
                previewText.textContent = 'Type a message to see preview...';
                previewText.classList.add('placeholder');
            }
        }

        // Update send button state
        const sendBtn = document.getElementById('sendBtn');
        if (sendBtn) {
            sendBtn.disabled = !text.trim();
        }

        // Update character counts
        this.updateTextStats(text);
    }

    updateTextStats(text) {
        const charCount = text.replace(/\s/g, '').length;
        const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
        
        document.getElementById('characterCount').textContent = `${charCount} characters`;
        document.getElementById('wordCount').textContent = `${wordCount} words`;
    }

    changeLanguage(language) {
        if (!this.isConnected || !this.socket) return;

        this.socket.emit('change_language', {
            language: language,
            user_type: this.userType
        });

        this.showNotification(`Language changed to ${this.getLanguageName(language)}`, 'success');
    }

    getLanguageName(code) {
        const languages = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil'
        };
        return languages[code] || code;
    }

    convertTextToSign(text) {
        if (!this.isConnected || !this.socket) {
            this.showError('Not connected to server');
            return;
        }

        this.socket.emit('convert_text_to_sign', {
            text: text,
            timestamp: Date.now()
        });
    }

    handleSignSequenceResult(data) {
        if (window.signDisplay) {
            window.signDisplay.showSignSequence(data.sign_sequence, data.original_text);
        }
    }

    handleSpeechResult(data) {
        if (data.success && data.text) {
            // Set the recognized text in input field
            const textInput = document.getElementById('textInput');
            if (textInput) {
                textInput.value = data.text;
                this.handleTextInput(data.text);
            }
            this.showNotification('Speech recognized successfully', 'success');
        } else {
            this.showError('Speech recognition failed');
        }
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
            const partnerName = this.userType === 'siva' ? 'Hari' : 'Siva';
            partnerStatus.textContent = status === 'connected' ? 
                `${partnerName} is online` : `Waiting for ${partnerName}...`;
        }
    }

    getCurrentTime() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    playNotificationSound() {
        // Simple notification sound using Web Audio API
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
        // Create and show notification
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

    showError(message) {
        this.showNotification(message, 'error');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Hari App Controller
class HariApp {
    constructor() {
        this.chatManager = null;
        this.speechRecognizer = null;
        this.isListening = false;
        this.sessionStartTime = null;
        this.timerInterval = null;
    }

    init() {
        try {
            // Initialize Chat Manager
            this.chatManager = new ChatManager('hari');
            this.chatManager.init();
            
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

        // Play preview button
        const playPreviewBtn = document.getElementById('playPreview');
        if (playPreviewBtn) {
            playPreviewBtn.addEventListener('click', () => this.playSignPreview());
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

        // Send message
        const success = this.chatManager.sendMessage(message, 'text');
        
        if (success) {
            // Convert to sign sequence for preview
            this.chatManager.convertTextToSign(message);
            
            // Clear input
            textInput.value = '';
            this.chatManager.handleTextInput('');
        }
    }

    useQuickResponse(text) {
        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.value = text;
            this.chatManager.handleTextInput(text);
        }
    }

    initSpeechRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn('Speech recognition not supported');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.speechRecognizer = new SpeechRecognition();
        
        this.speechRecognizer.continuous = false;
        this.speechRecognizer.interimResults = false;
        this.speechRecognizer.lang = 'en-US';

        this.speechRecognizer.onstart = () => {
            this.isListening = true;
            this.updateSpeechButton(true);
            this.chatManager.showNotification('Listening... Speak now', 'info');
        };

        this.speechRecognizer.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            const textInput = document.getElementById('textInput');
            if (textInput) {
                textInput.value = transcript;
                this.chatManager.handleTextInput(transcript);
            }
        };

        this.speechRecognizer.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.chatManager.showError(`Speech recognition error: ${event.error}`);
        };

        this.speechRecognizer.onend = () => {
            this.isListening = false;
            this.updateSpeechButton(false);
        };
    }

    toggleSpeechRecognition() {
        if (!this.speechRecognizer) {
            this.chatManager.showError('Speech recognition not supported');
            return;
        }

        if (this.isListening) {
            this.speechRecognizer.stop();
        } else {
            try {
                this.speechRecognizer.start();
            } catch (error) {
                this.chatManager.showError('Speech recognition failed to start');
            }
        }
    }

    updateSpeechButton(listening) {
        const speechBtn = document.getElementById('speechBtn');
        if (speechBtn) {
            if (listening) {
                speechBtn.classList.add('listening');
                speechBtn.innerHTML = '<i class="fas fa-stop"></i>';
            } else {
                speechBtn.classList.remove('listening');
                speechBtn.innerHTML = '<i class="fas fa-microphone"></i>';
            }
        }
    }

    playSignPreview() {
        if (window.signDisplay) {
            window.signDisplay.playAnimation();
        }
    }

    startSessionTimer() {
        this.sessionStartTime = Date.now();
        this.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.sessionStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
            const seconds = (elapsed % 60).toString().padStart(2, '0');
            document.getElementById('sessionTimer').textContent = `${minutes}:${seconds}`;
        }, 1000);
    }

    showError(message) {
        this.chatManager.showError(message);
    }

    destroy() {
        if (this.speechRecognizer) {
            this.speechRecognizer.stop();
        }
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }
        if (this.chatManager && this.chatManager.socket) {
            this.chatManager.socket.disconnect();
        }
    }
}