/**
 * COMPLETE WEBCAM DETECTION INTERFACE - Starts only when triggered
 * FIXED: Detection starts only when "Start Signing" is clicked
 */

class WebcamDetection {
    constructor() {
        this.isDetectionActive = false;
        this.currentText = '';
        this.rawText = '';
        this.processedText = '';
        this.formattedText = '';
        this.correctedText = '';
        this.handDetected = false;
        this.isAnalyzing = false;
        this.isInCooldown = false;
        this.lastPrediction = '';
        this.detectionCount = 0;
        this.targetLanguage = 'en';
        
        this.socket = io('http://172.20.10.4:5000', {
    transports: ['websocket', 'polling']
});
        this.setupSocketListeners();
        this.setupEventListeners();
        
        console.log('🎯 Webcam Detection System Initialized - Ready to start when triggered');
    }

    setupSocketListeners() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('✅ Connected to server');
            this.updateStatus('Connected to server', 'success');
            this.joinChat();
        });

        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from server');
            this.updateStatus('Disconnected from server', 'error');
        });

        this.socket.on('connect_error', (error) => {
            console.error('❌ Connection error:', error);
            this.updateStatus('Connection failed', 'error');
        });

        // Chat events
        this.socket.on('join_status', (data) => {
            if (data.status === 'joined') {
                console.log(`✅ Joined room: ${data.room_id}`);
                this.updatePartnerStatus('connected');
            }
        });

        this.socket.on('user_status', (data) => {
            if (data.user === 'hari') {
                this.updatePartnerStatus(data.status);
            }
        });

        this.socket.on('chat_history', (history) => {
            this.loadChatHistory(history);
        });

        // Detection events - TRIGGERED ONLY WHEN STARTED
        this.socket.on('detection_started', (data) => {
            console.log('🎬 Detection started:', data);
            this.isDetectionActive = true;
            this.updateStatus('Detection system started! OpenCV window should open.', 'success');
            this.updateDetectionUI();
            this.showNotification('Detection started! Look for the OpenCV window.', 'success');
        });

        this.socket.on('detection_stopped', (data) => {
            console.log('🛑 Detection stopped:', data);
            this.isDetectionActive = false;
            this.updateStatus('Detection system stopped', 'info');
            this.updateDetectionUI();
            this.showNotification('Detection stopped', 'info');
        });

        this.socket.on('detection_error', (data) => {
            console.error('❌ Detection error:', data);
            this.updateStatus(`Detection error: ${data.error}`, 'error');
            this.showNotification(`Detection error: ${data.error}`, 'error');
        });

        this.socket.on('detection_update', (data) => {
            this.handleDetectionUpdate(data);
        });

        this.socket.on('detection_status', (data) => {
            this.handleDetectionStatus(data);
        });

        // Text events
        this.socket.on('text_cleared', (data) => {
            console.log('🗑️ Text cleared:', data);
            this.clearAllText();
            this.updateStatus('Text cleared', 'info');
            this.showNotification('Text cleared', 'info');
        });

        this.socket.on('detection_reset', (data) => {
            console.log('🔄 Detection reset:', data);
            this.resetDetectionState();
            this.updateStatus('Detection reset', 'info');
            this.showNotification('Detection reset', 'info');
        });

        this.socket.on('language_set', (data) => {
            console.log('🌐 Language set:', data);
            this.targetLanguage = data.language;
            this.updateStatus(`Language set to ${this.getLanguageName(data.language)}`, 'success');
            this.showNotification(`Language set to ${this.getLanguageName(data.language)}`, 'success');
        });

        this.socket.on('text_processed', (data) => {
            console.log('🔤 Text processed:', data);
            this.updateStatus('Text processed through language pipeline', 'success');
        });

        this.socket.on('speech_started', (data) => {
            console.log('🔊 Speech started:', data);
            this.updateStatus('Speech synthesis started', 'info');
        });

        this.socket.on('message_sent', (data) => {
            console.log('💬 Message sent:', data);
            this.updateStatus('Message sent to Hari', 'success');
            this.showNotification('Message sent to Hari', 'success');
        });

        // Message events
        this.socket.on('receive_message', (data) => {
            this.handleIncomingMessage(data);
        });

        this.socket.on('message_sent_confirmation', (data) => {
            this.addChatMessage('siva', data.message, 'text');
        });

        console.log('📡 Socket listeners setup complete');
    }

    setupEventListeners() {
        // Detection control buttons
        const startBtn = document.getElementById('startDetection');
        const stopBtn = document.getElementById('stopDetection');
        const clearBtn = document.getElementById('clearText');
        const resetBtn = document.getElementById('resetDetection');
        const processBtn = document.getElementById('processText');
        const speakBtn = document.getElementById('speakText');
        const sendBtn = document.getElementById('sendMessage');

        if (startBtn) {
            startBtn.addEventListener('click', () => this.startDetection());
        }

        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopDetection());
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearText());
        }

        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetDetection());
        }

        if (processBtn) {
            processBtn.addEventListener('click', () => this.processText());
        }

        if (speakBtn) {
            speakBtn.addEventListener('click', () => this.speakText());
        }

        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessageToHari());
        }

        // Language selection
        const languageSelect = document.getElementById('languageSelect');
        if (languageSelect) {
            languageSelect.addEventListener('change', (e) => {
                this.setLanguage(e.target.value);
            });
        }

        // Start session timer
        this.startSessionTimer();

        console.log('🎮 Event listeners setup complete');
    }

    joinChat() {
        this.socket.emit('join_chat', {
            user_type: 'siva',
            room_id: 'siva_hari_chat'
        });
    }

    startDetection() {
        if (this.isDetectionActive) {
            this.showNotification('Detection already running', 'warning');
            return;
        }

        console.log('🚀 Starting detection on demand...');
        this.socket.emit('start_opencv_detection');
        this.updateStatus('Starting detection system...', 'info');
        this.showNotification('Starting detection... OpenCV window will open shortly.', 'info');
    }

    stopDetection() {
        if (!this.isDetectionActive) {
            this.showNotification('Detection not running', 'warning');
            return;
        }

        console.log('🛑 Stopping detection...');
        this.socket.emit('stop_opencv_detection');
        this.updateStatus('Stopping detection system...', 'info');
    }

    clearText() {
        console.log('🗑️ Clearing text...');
        this.socket.emit('clear_text');
    }

    resetDetection() {
        console.log('🔄 Resetting detection...');
        this.socket.emit('reset_detection');
    }

    setLanguage(language) {
        console.log(`🌐 Setting language to: ${language}`);
        this.socket.emit('set_language', { language });
    }

    processText() {
        console.log('🔤 Processing text...');
        this.socket.emit('process_text');
    }

    speakText() {
        console.log('🔊 Speaking text...');
        this.socket.emit('speak_text');
    }

    sendMessageToHari() {
        if (!this.processedText && !this.currentText) {
            this.showNotification('No text to send', 'error');
            return;
        }

        console.log('💬 Sending message to Hari...');
        this.socket.emit('send_message_to_hari');
        
        // Also send via chat for consistency
        const messageToSend = this.processedText || this.currentText;
        this.socket.emit('send_chat_message', {
            message: messageToSend,
            user_type: 'siva',
            target_language: this.targetLanguage
        });
    }

    handleDetectionUpdate(data) {
        console.log('📊 Detection update:', data);
        
        // Update all text fields
        if (data.current_text !== undefined) this.currentText = data.current_text;
        if (data.raw_text !== undefined) this.rawText = data.raw_text;
        if (data.processed_text !== undefined) this.processedText = data.processed_text;
        if (data.formatted_text !== undefined) this.formattedText = data.formatted_text;
        if (data.corrected_text !== undefined) this.correctedText = data.corrected_text;
        
        if (data.prediction) {
            this.lastPrediction = data.prediction;
            this.detectionCount++;
            
            // Update prediction display
            this.updatePredictionDisplay(data.prediction, data.confidence);
        }
        
        this.updateTextDisplay();
        this.updateDetectionUI();
    }

    handleDetectionStatus(data) {
        console.log('📈 Detection status:', data);
        
        this.handDetected = data.hand_detected || false;
        this.isAnalyzing = data.is_analyzing || false;
        this.isInCooldown = data.is_in_cooldown || false;
        
        if (data.last_prediction !== undefined) this.lastPrediction = data.last_prediction;
        if (data.detection_count !== undefined) this.detectionCount = data.detection_count;
        if (data.current_text !== undefined) this.currentText = data.current_text;
        if (data.raw_text !== undefined) this.rawText = data.raw_text;
        if (data.processed_text !== undefined) this.processedText = data.processed_text;
        if (data.formatted_text !== undefined) this.formattedText = data.formatted_text;
        if (data.corrected_text !== undefined) this.correctedText = data.corrected_text;
        if (data.target_language !== undefined) this.targetLanguage = data.target_language;
        
        this.updateDetectionUI();
        this.updateTextDisplay();
    }

    handleIncomingMessage(data) {
        if (data.from === 'hari') {
            this.addChatMessage('hari', data.message, 'text');
            
            // Display sign sequence if available
            if (data.sign_sequence && window.signDisplay) {
                window.signDisplay.showSignSequence(data.sign_sequence, data.message);
            }
            
            // Play notification sound
            this.playNotificationSound();
            
            this.showNotification(`New message from Hari: ${data.message}`, 'info');
        }
    }

    updateDetectionUI() {
        // Update status indicators
        this.updateElement('detectionStatus', this.isDetectionActive ? '🟢 Active' : '🔴 Inactive');
        this.updateElement('handStatus', this.handDetected ? '🟢 Detected' : '🔴 Not Detected');
        this.updateElement('lastPrediction', this.lastPrediction || '-');
        this.updateElement('detectionState', this.getDetectionStateText());
        this.updateElement('detectionCount', `Detections: ${this.detectionCount}`);
        
        // Update confidence (if available)
        const confidenceElement = document.getElementById('confidence');
        if (confidenceElement) {
            confidenceElement.textContent = this.lastPrediction ? 'High' : '-';
        }
        
        // Update button states
        this.updateButtonState('startDetection', !this.isDetectionActive);
        this.updateButtonState('stopDetection', this.isDetectionActive);
        this.updateButtonState('sendMessage', !!(this.processedText || this.currentText));
        
        // Update button texts based on state
        const startBtn = document.getElementById('startDetection');
        if (startBtn) {
            startBtn.innerHTML = this.isDetectionActive ? 
                '<i class="fas fa-sync-alt fa-spin"></i> Detection Running...' : 
                '<i class="fas fa-play"></i> Start Signing';
        }
        
        const stopBtn = document.getElementById('stopDetection');
        if (stopBtn) {
            stopBtn.innerHTML = this.isDetectionActive ? 
                '<i class="fas fa-stop"></i> Stop Signing' : 
                '<i class="fas fa-stop"></i> Stop (Inactive)';
        }
    }

    getDetectionStateText() {
        if (this.isAnalyzing) return '🔄 Analyzing';
        if (this.isInCooldown) return '⏳ Cooldown';
        if (this.handDetected) return '✅ Hand Detected';
        return '🔵 Ready';
    }

    updateTextDisplay() {
        // Update main text display
        const textDisplay = document.getElementById('currentTextDisplay');
        if (textDisplay) {
            const displayText = this.processedText || this.currentText || this.rawText;
            if (displayText) {
                textDisplay.innerHTML = `
                    <div class="detected-text">
                        <div class="text-content">${this.escapeHtml(displayText)}</div>
                        ${this.correctedText && this.correctedText !== displayText ? 
                         `<div class="corrected-text">Corrected: ${this.escapeHtml(this.correctedText)}</div>` : ''}
                        ${this.formattedText && this.formattedText !== displayText ? 
                         `<div class="formatted-text">Formatted: ${this.escapeHtml(this.formattedText)}</div>` : ''}
                    </div>
                `;
            } else {
                textDisplay.innerHTML = `
                    <div class="placeholder-text">
                        <i class="fas fa-keyboard"></i>
                        <p>Click "Start Signing" to begin detection...</p>
                    </div>
                `;
            }
        }
        
        // Update stats
        const displayText = this.processedText || this.currentText || this.rawText || '';
        const charCount = displayText.replace(/\s/g, '').length;
        const wordCount = displayText.trim() ? displayText.trim().split(/\s+/).length : 0;
        
        this.updateElement('characterCount', `${charCount} characters`);
        this.updateElement('wordCount', `${wordCount} words`);
        
        // Update send button
        this.updateButtonState('sendMessage', !!displayText);
    }

    updatePredictionDisplay(prediction, confidence) {
        const predictionElement = document.getElementById('lastPrediction');
        if (predictionElement) {
            predictionElement.textContent = prediction;
        }
        
        const confidenceElement = document.getElementById('confidence');
        if (confidenceElement && confidence) {
            confidenceElement.textContent = `${(confidence * 100).toFixed(1)}%`;
        }
        
        // Show temporary notification
        this.showNotification(`Detected: ${prediction} (${(confidence * 100).toFixed(1)}%)`, 'info', 2000);
    }

    clearAllText() {
        this.currentText = '';
        this.rawText = '';
        this.processedText = '';
        this.formattedText = '';
        this.correctedText = '';
        this.updateTextDisplay();
    }

    resetDetectionState() {
        this.handDetected = false;
        this.isAnalyzing = false;
        this.isInCooldown = false;
        this.lastPrediction = '';
        this.updateDetectionUI();
    }

    addChatMessage(sender, message, type = 'text') {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        messageContent.innerHTML = `
            <div class="message-header">
                <strong>${sender === 'siva' ? 'Siva' : 'Hari'}</strong>
                <span class="message-time">${this.getCurrentTime()}</span>
            </div>
            <div class="message-text">${this.escapeHtml(message)}</div>
        `;
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
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

    updateStatus(message, type = 'info') {
        const statusElement = document.getElementById('systemStatus');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status-${type}`;
        }
        
        // Add to log
        this.addToLog(message, type);
    }

    updatePartnerStatus(status) {
        const partnerStatus = document.getElementById('partnerStatus');
        if (partnerStatus) {
            partnerStatus.textContent = status === 'connected' ? 
                'Hari is online' : 'Waiting for Hari...';
        }
    }

    addToLog(message, type = 'info') {
        const logElement = document.getElementById('detectionLog');
        if (logElement) {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry log-${type}`;
            logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> ${message}`;
            
            logElement.appendChild(logEntry);
            logElement.scrollTop = logElement.scrollHeight;
        }
    }

    showNotification(message, type = 'info', duration = 3000) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Show with animation
        setTimeout(() => notification.classList.add('show'), 10);
        
        // Remove after duration
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, duration);
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

    getLanguageName(code) {
        const languages = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil'
        };
        return languages[code] || code;
    }

    getCurrentTime() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    updateElement(id, content) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = content;
        }
    }

    updateButtonState(buttonId, enabled) {
        const button = document.getElementById(buttonId);
        if (button) {
            button.disabled = !enabled;
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Get current detection state
    getDetectionState() {
        return {
            isDetectionActive: this.isDetectionActive,
            currentText: this.currentText,
            rawText: this.rawText,
            processedText: this.processedText,
            formattedText: this.formattedText,
            correctedText: this.correctedText,
            handDetected: this.handDetected,
            isAnalyzing: this.isAnalyzing,
            isInCooldown: this.isInCooldown,
            lastPrediction: this.lastPrediction,
            detectionCount: this.detectionCount,
            targetLanguage: this.targetLanguage
        };
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.webcamDetection = new WebcamDetection();
    console.log('🎯 Webcam Detection System Ready - Click "Start Signing" to begin');
    
    // Initialize sign display if available
    if (typeof SignDisplay !== 'undefined') {
        window.signDisplay = new SignDisplay('signDisplayContainer');
        console.log('🔤 Sign Display System Ready');
    }
});