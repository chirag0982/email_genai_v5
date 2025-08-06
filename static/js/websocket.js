// WebSocket Manager for Real-time Features

class WebSocketManager {
    constructor() {
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 3000;
        this.connectionPromise = null;
        this.messageHandlers = new Map();
        this.isAuthenticated = false;
        
        // Bind methods
        this.connect = this.connect.bind(this);
        this.disconnect = this.disconnect.bind(this);
        this.emit = this.emit.bind(this);
        this.on = this.on.bind(this);
        this.off = this.off.bind(this);
    }
    
    async connect() {
        if (this.socket && this.socket.connected) {
            return this.socket;
        }
        
        if (this.connectionPromise) {
            return this.connectionPromise;
        }
        
        this.connectionPromise = new Promise((resolve, reject) => {
            try {
                // Initialize Socket.IO connection
                this.socket = io({
                    transports: ['websocket', 'polling'],
                    timeout: 10000,
                    reconnection: true,
                    reconnectionDelay: 1000,
                    reconnectionAttempts: this.maxReconnectAttempts
                });
                
                // Connection event handlers
                this.socket.on('connect', () => {
                    console.log('WebSocket connected:', this.socket.id);
                    this.reconnectAttempts = 0;
                    this.isAuthenticated = true;
                    this.setupEventHandlers();
                    resolve(this.socket);
                });
                
                this.socket.on('disconnect', (reason) => {
                    console.log('WebSocket disconnected:', reason);
                    this.isAuthenticated = false;
                    
                    if (reason === 'io server disconnect') {
                        // Server initiated disconnect, attempt to reconnect
                        this.attemptReconnect();
                    }
                });
                
                this.socket.on('connect_error', (error) => {
                    console.error('WebSocket connection error:', error);
                    this.isAuthenticated = false;
                    reject(error);
                });
                
                this.socket.on('reconnect', (attemptNumber) => {
                    console.log('WebSocket reconnected after', attemptNumber, 'attempts');
                    this.reconnectAttempts = 0;
                    this.isAuthenticated = true;
                    this.showConnectionStatus('Reconnected to server', 'success');
                });
                
                this.socket.on('reconnect_error', (error) => {
                    console.error('WebSocket reconnection error:', error);
                    this.handleReconnectError();
                });
                
                this.socket.on('reconnect_failed', () => {
                    console.error('WebSocket reconnection failed after maximum attempts');
                    this.showConnectionStatus('Connection lost. Please refresh the page.', 'error');
                });
                
            } catch (error) {
                console.error('Failed to initialize WebSocket:', error);
                reject(error);
            }
        });
        
        return this.connectionPromise;
    }
    
    setupEventHandlers() {
        if (!this.socket) return;
        
        // System events
        this.socket.on('connected', (data) => {
            console.log('WebSocket authenticated:', data);
            this.showConnectionStatus('Connected to real-time server', 'success');
        });
        
        this.socket.on('error', (data) => {
            console.error('WebSocket error:', data);
            this.showConnectionStatus('Server error: ' + data.message, 'error');
        });
        
        // Collaboration events
        this.socket.on('user_joined', (data) => {
            this.handleUserJoined(data);
        });
        
        this.socket.on('user_left', (data) => {
            this.handleUserLeft(data);
        });
        
        this.socket.on('content_updated', (data) => {
            this.handleContentUpdated(data);
        });
        
        this.socket.on('cursor_moved', (data) => {
            this.handleCursorMoved(data);
        });
        
        // AI events
        this.socket.on('ai_generation_started', (data) => {
            this.handleAIGenerationStarted(data);
        });
        
        this.socket.on('ai_generation_completed', (data) => {
            this.handleAIGenerationCompleted(data);
        });
        
        // Email events
        this.socket.on('email_sent_notification', (data) => {
            this.handleEmailSentNotification(data);
        });
        
        // Setup ping/pong for keepalive
        this.setupPingPong();
    }
    
    setupPingPong() {
        setInterval(() => {
            if (this.socket && this.socket.connected) {
                this.socket.emit('ping');
            }
        }, 30000); // Ping every 30 seconds
        
        this.socket.on('pong', () => {
            // Connection is alive
        });
    }
    
    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        this.connectionPromise = null;
        this.isAuthenticated = false;
    }
    
    emit(event, data = {}) {
        if (!this.socket || !this.socket.connected) {
            console.warn('WebSocket not connected, queuing event:', event);
            this.connect().then(() => {
                this.socket.emit(event, data);
            }).catch(error => {
                console.error('Failed to emit event:', event, error);
            });
            return;
        }
        
        this.socket.emit(event, data);
    }
    
    on(event, handler) {
        this.messageHandlers.set(event, handler);
        if (this.socket) {
            this.socket.on(event, handler);
        }
    }
    
    off(event, handler) {
        this.messageHandlers.delete(event);
        if (this.socket) {
            this.socket.off(event, handler);
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.showConnectionStatus('Unable to reconnect. Please refresh the page.', 'error');
            return;
        }
        
        this.reconnectAttempts++;
        this.showConnectionStatus(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`, 'warning');
        
        setTimeout(() => {
            this.connect().catch(error => {
                console.error('Reconnection attempt failed:', error);
            });
        }, this.reconnectInterval);
    }
    
    handleReconnectError() {
        this.reconnectAttempts++;
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.showConnectionStatus(`Connection lost. Retrying... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`, 'warning');
        }
    }
    
    showConnectionStatus(message, type) {
        if (window.appUtils && window.appUtils.showToast) {
            window.appUtils.showToast(message, type, type === 'error' ? 0 : 3000);
        }
    }
    
    // Collaboration event handlers
    handleUserJoined(data) {
        console.log('User joined collaboration:', data);
        
        // Update collaboration indicator
        this.updateCollaborationIndicator(data.active_users);
        
        // Show notification
        if (data.user_id !== this.getCurrentUserId()) {
            this.showConnectionStatus(`${data.user_name} joined the collaboration`, 'info');
        }
    }
    
    handleUserLeft(data) {
        console.log('User left collaboration:', data);
        
        // Update collaboration indicator
        this.updateCollaborationIndicator(data.active_users);
        
        // Remove user's cursor
        this.removeCursor(data.user_id);
    }
    
    handleContentUpdated(data) {
        // Don't update if it's from the current user
        if (data.user_id === this.getCurrentUserId()) {
            return;
        }
        
        console.log('Content updated by another user:', data);
        
        // Update content in the editor
        const emailBody = document.getElementById('emailBody');
        if (emailBody && data.content) {
            const currentCursor = emailBody.selectionStart;
            emailBody.value = data.content;
            
            // Try to maintain cursor position
            emailBody.setSelectionRange(currentCursor, currentCursor);
            
            // Show subtle indicator
            this.showContentUpdateIndicator();
        }
    }
    
    handleCursorMoved(data) {
        if (data.user_id === this.getCurrentUserId()) {
            return;
        }
        
        console.log('Cursor moved:', data);
        this.updateCursor(data.user_id, data.cursor_position);
    }
    
    handleAIGenerationStarted(data) {
        if (data.user_id === this.getCurrentUserId()) {
            return;
        }
        
        console.log('AI generation started by another user:', data);
        this.showConnectionStatus(`AI is generating a response using ${data.model}...`, 'info');
    }
    
    handleAIGenerationCompleted(data) {
        if (data.user_id === this.getCurrentUserId()) {
            return;
        }
        
        console.log('AI generation completed by another user:', data);
        this.showConnectionStatus(`AI response generated in ${(data.generation_time_ms / 1000).toFixed(1)}s`, 'success');
    }
    
    handleEmailSentNotification(data) {
        console.log('Email sent notification:', data);
        this.showConnectionStatus(`Email "${data.subject}" sent by ${data.sender}`, 'success');
    }
    
    // UI helper methods
    updateCollaborationIndicator(activeUsers) {
        const indicator = document.getElementById('collaborationIndicator');
        const count = document.getElementById('collaboratorCount');
        
        if (indicator && count) {
            const userCount = activeUsers ? activeUsers.length : 0;
            
            if (userCount > 1) {
                indicator.classList.remove('d-none');
                count.textContent = userCount - 1; // Exclude current user
                
                // Update collaborators list
                this.updateCollaboratorsList(activeUsers);
            } else {
                indicator.classList.add('d-none');
            }
        }
    }
    
    updateCollaboratorsList(activeUsers) {
        const collaboratorsDiv = document.getElementById('activeCollaborators');
        if (!collaboratorsDiv || !activeUsers) return;
        
        const currentUserId = this.getCurrentUserId();
        const otherUsers = activeUsers.filter(user => user.user_id !== currentUserId);
        
        collaboratorsDiv.innerHTML = otherUsers.map(user => `
            <div class="d-flex align-items-center mb-2">
                ${user.user_image ? 
                    `<img src="${user.user_image}" alt="${user.user_name}" class="rounded-circle me-2" width="24" height="24" style="object-fit: cover;">` :
                    `<div class="bg-primary bg-opacity-25 rounded-circle me-2 d-flex align-items-center justify-content-center" style="width: 24px; height: 24px;">
                        <i data-feather="user" style="width: 12px; height: 12px;"></i>
                    </div>`
                }
                <div>
                    <div class="small fw-medium">${user.user_name}</div>
                    <div class="text-muted" style="font-size: 11px;">Active</div>
                </div>
            </div>
        `).join('');
        
        // Re-initialize feather icons
        if (window.feather) {
            feather.replace();
        }
    }
    
    updateCursor(userId, position) {
        // Remove existing cursor for this user
        this.removeCursor(userId);
        
        // Add new cursor indicator
        const emailBody = document.getElementById('emailBody');
        if (emailBody) {
            const cursor = document.createElement('div');
            cursor.id = `cursor-${userId}`;
            cursor.className = 'collaborator-cursor';
            cursor.style.borderColor = this.getUserColor(userId);
            cursor.setAttribute('data-user', this.getUserName(userId));
            
            // Position cursor (simplified - in a real app you'd calculate exact position)
            cursor.style.left = '10px';
            cursor.style.top = `${position * 0.5}px`; // Rough approximation
            
            emailBody.parentElement.appendChild(cursor);
        }
    }
    
    removeCursor(userId) {
        const cursor = document.getElementById(`cursor-${userId}`);
        if (cursor) {
            cursor.remove();
        }
    }
    
    showContentUpdateIndicator() {
        const emailBody = document.getElementById('emailBody');
        if (emailBody) {
            emailBody.style.borderColor = '#20c997';
            emailBody.style.boxShadow = '0 0 0 0.2rem rgba(32, 201, 151, 0.25)';
            
            setTimeout(() => {
                emailBody.style.borderColor = '';
                emailBody.style.boxShadow = '';
            }, 1000);
        }
    }
    
    // Utility methods
    getCurrentUserId() {
        // This would typically come from the authenticated user data
        return window.currentUser?.id || null;
    }
    
    getUserColor(userId) {
        // Generate consistent color for user
        const colors = ['#0d6efd', '#20c997', '#ffc107', '#dc3545', '#6f42c1'];
        const index = userId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
        return colors[index % colors.length];
    }
    
    getUserName(userId) {
        // This would typically come from a user cache
        return `User ${userId.substring(0, 6)}`;
    }
    
    // Public API
    isConnected() {
        return this.socket && this.socket.connected && this.isAuthenticated;
    }
    
    getConnectionState() {
        if (!this.socket) return 'disconnected';
        if (this.socket.connected && this.isAuthenticated) return 'connected';
        if (this.socket.connecting) return 'connecting';
        return 'disconnected';
    }
}

// Create global WebSocket manager instance
let wsManager = null;

// Initialize WebSocket when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize if user is authenticated
    if (document.body.dataset.authenticated === 'true') {
        initializeWebSocket();
    }
});

function initializeWebSocket() {
    if (!wsManager) {
        wsManager = new WebSocketManager();
        
        // Connect automatically
        wsManager.connect().catch(error => {
            console.error('Failed to initialize WebSocket connection:', error);
        });
        
        // Make available globally
        window.socket = wsManager;
    }
    
    return wsManager;
}

// Export for use in other modules
window.WebSocketManager = WebSocketManager;
window.initializeWebSocket = initializeWebSocket;
