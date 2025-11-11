/**
 * UPDATED SIGN DISPLAY MANAGER
 * FIXED: Sign image loading and display
 * FIXED: Animation controls and progress tracking
 * FIXED: Real-time sign sequence updates
 */

class SignDisplayManager {
    constructor() {
        this.currentSequence = [];
        this.isAnimating = false;
        this.animationSpeed = 1000; // ms per sign
        this.currentSignIndex = 0;
        this.animationInterval = null;
        this.container = null;
        this.originalText = '';
        this.correctedText = '';
        
        // Cache for loaded images
        this.imageCache = new Map();
        
        this.initialize();
    }
    
    initialize() {
        console.log('🔄 Initializing Sign Display Manager...');
        this.container = document.getElementById('signDisplayContainer');
        
        if (!this.container) {
            console.error('❌ Sign display container not found');
            return;
        }
        
        this.setupEventListeners();
        this.showPlaceholder();
    }
    
    setupEventListeners() {
        // Socket event listeners
        if (typeof socket !== 'undefined') {
            socket.on('sign_sequence_update', (data) => {
                console.log('🔄 Received sign sequence update:', data);
                this.updateSignSequence(data.sign_sequence, data.original_text, data.corrected_text);
            });
            
            socket.on('detection_update', (data) => {
                if (data.processed_text) {
                    console.log('🔄 Received detection update:', data.processed_text);
                    // Convert text to sign sequence
                    const sequence = this.textToSignSequence(data.processed_text);
                    this.updateSignSequence(sequence, data.original_text, data.corrected_text);
                }
            });
        }
        
        // Animation control listeners
        const playBtn = document.getElementById('playAnimation');
        const stopBtn = document.getElementById('stopAnimation');
        const speedSelect = document.getElementById('animationSpeed');
        
        if (playBtn) {
            playBtn.addEventListener('click', () => this.startAnimation());
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopAnimation());
        }
        
        if (speedSelect) {
            speedSelect.addEventListener('change', (e) => {
                this.setAnimationSpeed(parseInt(e.target.value));
            });
        }
    }
    
    textToSignSequence(text) {
        if (!text) return [];
        
        console.log(`🔤 Converting text to sign sequence: "${text}"`);
        
        const sequence = [];
        const words = text.toUpperCase().split(' ');
        
        for (let i = 0; i < words.length; i++) {
            const word = words[i];
            
            // Add each letter of the word
            for (let j = 0; j < word.length; j++) {
                const char = word[j];
                if (/[A-Z]/.test(char)) {
                    sequence.push(char);
                }
            }
            
            // Add space between words (but not after the last word)
            if (i < words.length - 1) {
                sequence.push('SPACE');
            }
        }
        
        console.log(`✅ Generated sequence:`, sequence);
        return sequence;
    }
    
    updateSignSequence(sequence, originalText = '', correctedText = '') {
        console.log(`🎯 Updating sign sequence:`, sequence);
        
        this.currentSequence = Array.isArray(sequence) ? sequence : [];
        this.originalText = originalText || '';
        this.correctedText = correctedText || '';
        this.currentSignIndex = 0;
        
        this.stopAnimation();
        this.renderSequence();
        
        // Update statistics
        this.updateStatistics();
    }
    
    async renderSequence() {
        if (!this.container) return;
        
        if (this.currentSequence.length === 0) {
            this.showPlaceholder();
            return;
        }
        
        console.log(`🎨 Rendering ${this.currentSequence.length} signs`);
        
        // Clear container
        this.container.innerHTML = '';
        
        // Add original text display
        if (this.originalText) {
            const originalTextDiv = document.createElement('div');
            originalTextDiv.className = 'original-text-enhanced';
            originalTextDiv.innerHTML = `
                <strong>Original Text:</strong> "${this.originalText}"
                ${this.correctedText && this.correctedText !== this.originalText ? 
                    `<br><strong>Corrected Text:</strong> "${this.correctedText}"` : ''}
            `;
            this.container.appendChild(originalTextDiv);
        }
        
        // Create sign sequence container
        const sequenceContainer = document.createElement('div');
        sequenceContainer.className = 'sign-sequence-enhanced';
        
        // Group signs into rows (max 5 signs per row for better display)
        const signsPerRow = 5;
        const rows = [];
        
        for (let i = 0; i < this.currentSequence.length; i += signsPerRow) {
            rows.push(this.currentSequence.slice(i, i + signsPerRow));
        }
        
        // Create rows
        for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
            const row = rows[rowIndex];
            const rowDiv = document.createElement('div');
            rowDiv.className = 'sign-row-enhanced';
            
            for (let signIndex = 0; signIndex < row.length; signIndex++) {
                const signClass = row[signIndex];
                const globalIndex = rowIndex * signsPerRow + signIndex;
                
                const signElement = await this.createSignElement(signClass, globalIndex);
                rowDiv.appendChild(signElement);
            }
            
            sequenceContainer.appendChild(rowDiv);
        }
        
        this.container.appendChild(sequenceContainer);
        
        // Add animation controls if we have signs
        if (this.currentSequence.length > 0) {
            this.addAnimationControls();
        }
        
        console.log(`✅ Sign sequence rendered successfully`);
    }
    
    async createSignElement(signClass, index) {
        const signDiv = document.createElement('div');
        signDiv.className = 'sign-element-enhanced';
        signDiv.dataset.signClass = signClass;
        signDiv.dataset.index = index;
        
        // Create loading state
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'sign-loading';
        loadingDiv.innerHTML = `
            <i class="fas fa-spinner fa-spin"></i>
            <span>Loading...</span>
        `;
        signDiv.appendChild(loadingDiv);
        
        try {
            // Load sign image
            const imgUrl = `/api/sign-image/${signClass}`;
            console.log(`🖼️ Loading sign image: ${imgUrl}`);
            
            const img = await this.loadSignImage(imgUrl, signClass);
            signDiv.innerHTML = ''; // Clear loading
            signDiv.appendChild(img);
            
            // Add label
            const labelDiv = document.createElement('div');
            labelDiv.className = 'sign-label-enhanced';
            labelDiv.textContent = this.formatSignLabel(signClass);
            signDiv.appendChild(labelDiv);
            
        } catch (error) {
            console.error(`❌ Failed to load sign image for ${signClass}:`, error);
            signDiv.innerHTML = '';
            signDiv.className = 'sign-element-enhanced sign-error';
            signDiv.innerHTML = `
                <div style="text-align: center; padding: 10px;">
                    <i class="fas fa-exclamation-triangle"></i>
                    <div>${signClass}</div>
                    <small>Failed to load</small>
                </div>
            `;
        }
        
        return signDiv;
    }
    
    async loadSignImage(url, signClass) {
        // Check cache first
        if (this.imageCache.has(signClass)) {
            console.log(`📦 Using cached image for: ${signClass}`);
            return this.imageCache.get(signClass).cloneNode();
        }
        
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                console.log(`✅ Image loaded successfully: ${signClass}`);
                // Cache the image
                this.imageCache.set(signClass, img);
                resolve(img);
            };
            
            img.onerror = (error) => {
                console.error(`❌ Image load error for ${signClass}:`, error);
                reject(new Error(`Failed to load image for ${signClass}`));
            };
            
            // Add cache busting to prevent browser caching issues
            const cacheBustUrl = `${url}?t=${Date.now()}`;
            console.log(`📥 Loading image from: ${cacheBustUrl}`);
            img.src = cacheBustUrl;
            img.alt = `Sign for ${signClass}`;
            img.style.width = '100%';
            img.style.height = '100%';
            img.style.objectFit = 'cover';
        });
    }
    
    formatSignLabel(signClass) {
        switch (signClass) {
            case 'SPACE': return 'Space';
            case 'ENTER': return 'Enter';
            case 'BACKSPACE': return 'Backspace';
            default: return signClass;
        }
    }
    
    addAnimationControls() {
        // Remove existing controls
        const existingControls = this.container.querySelector('.animation-controls');
        if (existingControls) {
            existingControls.remove();
        }
        
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'animation-controls';
        controlsDiv.innerHTML = `
            <button id="playAnimation" class="btn-primary btn-play">
                <i class="fas fa-play"></i> Play Animation
            </button>
            <button id="stopAnimation" class="btn-secondary btn-stop">
                <i class="fas fa-stop"></i> Stop
            </button>
            <select id="animationSpeed" class="animation-speed-select">
                <option value="2000">Slow</option>
                <option value="1000" selected>Normal</option>
                <option value="500">Fast</option>
                <option value="250">Very Fast</option>
            </select>
            <div class="animation-progress">
                <div class="progress-fill" id="animationProgress"></div>
            </div>
        `;
        
        this.container.appendChild(controlsDiv);
        
        // Add event listeners to new controls
        document.getElementById('playAnimation').addEventListener('click', () => this.startAnimation());
        document.getElementById('stopAnimation').addEventListener('click', () => this.stopAnimation());
        document.getElementById('animationSpeed').addEventListener('change', (e) => {
            this.setAnimationSpeed(parseInt(e.target.value));
        });
    }
    
    startAnimation() {
        if (this.isAnimating || this.currentSequence.length === 0) {
            return;
        }
        
        console.log('🎬 Starting animation');
        this.isAnimating = true;
        this.currentSignIndex = 0;
        
        // Update button states
        this.updateAnimationControls();
        
        // Start animation interval
        this.animationInterval = setInterval(() => {
            this.highlightCurrentSign();
            this.currentSignIndex++;
            
            // Update progress
            this.updateProgress();
            
            if (this.currentSignIndex >= this.currentSequence.length) {
                this.stopAnimation();
                // Optionally restart or stop
                setTimeout(() => {
                    this.resetHighlight();
                    this.currentSignIndex = 0;
                    this.updateProgress();
                }, 500);
            }
        }, this.animationSpeed);
        
        // Highlight first sign immediately
        this.highlightCurrentSign();
        this.updateProgress();
    }
    
    stopAnimation() {
        if (this.animationInterval) {
            clearInterval(this.animationInterval);
            this.animationInterval = null;
        }
        
        this.isAnimating = false;
        this.resetHighlight();
        this.updateAnimationControls();
        console.log('🛑 Animation stopped');
    }
    
    setAnimationSpeed(speed) {
        this.animationSpeed = speed;
        console.log(`⚡ Animation speed set to: ${speed}ms`);
        
        // Restart animation if it's running
        if (this.isAnimating) {
            this.stopAnimation();
            this.startAnimation();
        }
    }
    
    highlightCurrentSign() {
        // Remove highlight from all signs
        this.resetHighlight();
        
        // Highlight current sign
        if (this.currentSignIndex < this.currentSequence.length) {
            const currentSign = this.container.querySelector(`[data-index="${this.currentSignIndex}"]`);
            if (currentSign) {
                currentSign.classList.add('highlighted');
                
                // Scroll to center the highlighted sign
                currentSign.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center',
                    inline: 'center'
                });
            }
        }
    }
    
    resetHighlight() {
        const highlightedSigns = this.container.querySelectorAll('.highlighted');
        highlightedSigns.forEach(sign => {
            sign.classList.remove('highlighted');
        });
    }
    
    updateProgress() {
        const progressFill = document.getElementById('animationProgress');
        if (progressFill) {
            const progress = this.currentSequence.length > 0 
                ? (this.currentSignIndex / this.currentSequence.length) * 100 
                : 0;
            progressFill.style.width = `${progress}%`;
        }
    }
    
    updateAnimationControls() {
        const playBtn = document.getElementById('playAnimation');
        const stopBtn = document.getElementById('stopAnimation');
        
        if (playBtn) {
            playBtn.disabled = this.isAnimating || this.currentSequence.length === 0;
        }
        
        if (stopBtn) {
            stopBtn.disabled = !this.isAnimating;
        }
    }
    
    updateStatistics() {
        const stats = {
            totalSigns: this.currentSequence.length,
            uniqueSigns: new Set(this.currentSequence).size,
            hasSpace: this.currentSequence.includes('SPACE'),
            hasEnter: this.currentSequence.includes('ENTER'),
            hasBackspace: this.currentSequence.includes('BACKSPACE')
        };
        
        console.log('📊 Sign statistics:', stats);
        
        // Update UI statistics if elements exist
        const totalSignsEl = document.getElementById('totalSigns');
        const uniqueSignsEl = document.getElementById('uniqueSigns');
        
        if (totalSignsEl) totalSignsEl.textContent = stats.totalSigns;
        if (uniqueSignsEl) uniqueSignsEl.textContent = stats.uniqueSigns;
    }
    
    showPlaceholder() {
        if (!this.container) return;
        
        this.container.innerHTML = `
            <div class="placeholder-signs">
                <i class="fas fa-hands"></i>
                <p>No sign sequence to display</p>
                <small>Signs will appear here when Hari sends a message or when you start signing</small>
            </div>
        `;
        
        // Remove animation controls
        const existingControls = this.container.querySelector('.animation-controls');
        if (existingControls) {
            existingControls.remove();
        }
        
        // Reset progress
        this.updateProgress();
    }
    
    clearSequence() {
        this.currentSequence = [];
        this.originalText = '';
        this.correctedText = '';
        this.stopAnimation();
        this.showPlaceholder();
        this.updateStatistics();
    }
    
    // Public method to manually set sequence
    setSequence(sequence, originalText = '', correctedText = '') {
        this.updateSignSequence(sequence, originalText, correctedText);
    }
    
    // Public method to get current sequence
    getCurrentSequence() {
        return {
            sequence: this.currentSequence,
            originalText: this.originalText,
            correctedText: this.correctedText,
            isAnimating: this.isAnimating,
            currentIndex: this.currentSignIndex
        };
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.signDisplayManager = new SignDisplayManager();
    console.log('✅ Sign Display Manager initialized');
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SignDisplayManager;
}