// Collaboration Features for Email Composition

class CollaborationManager {
    constructor(emailId, websocketManager) {
        this.emailId = emailId;
        this.ws = websocketManager;
        this.isActive = false;
        this.collaborators = new Map();
        this.pendingChanges = [];
        this.lastSyncTime = Date.now();
        this.contentHistory = [];
        
        // Debounced functions
        this.debouncedSendCursorUpdate = this.debounce(this.sendCursorUpdate.bind(this), 100);
        this.debouncedSendContentUpdate = this.debounce(this.sendContentUpdate.bind(this), 500);
        
        this.initializeCollaboration();
    }
    
    initializeCollaboration() {
        if (!this.emailId || !this.ws) {
            console.warn('Cannot initialize collaboration: missing emailId or websocket');
            return;
        }
        
        this.setupEventListeners();
        this.joinCollaboration();
    }
    
    setupEventListeners() {
        // Content change tracking
        const emailBody = document.getElementById('emailBody');
        const emailSubject = document.getElementById('emailSubject');
        
        if (emailBody) {
            emailBody.addEventListener('input', (e) => {
                this.handleContentChange(e, 'body');
            });
            
            emailBody.addEventListener('selectionchange', (e) => {
                this.handleCursorChange(e);
            });
            
            emailBody.addEventListener('keyup', (e) => {
                this.handleCursorChange(e);
            });
            
            emailBody.addEventListener('click', (e) => {
                this.handleCursorChange(e);
            });
        }
        
        if (emailSubject) {
            emailSubject.addEventListener('input', (e) => {
                this.handleContentChange(e, 'subject');
            });
        }
        
        // Window/tab events
        window.addEventListener('beforeunload', () => {
            this.leaveCollaboration();
        });
        
        // Visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseCollaboration();
            } else {
                this.resumeCollaboration();
            }
        });
        
        // WebSocket events
        if (this.ws) {
            this.ws.on('collaboration_joined', (data) => {
                this.handleCollaborationJoined(data);
            });
            
            this.ws.on('user_joined', (data) => {
                this.handleUserJoined(data);
            });
            
            this.ws.on('user_left', (data) => {
                this.handleUserLeft(data);
            });
            
            this.ws.on('content_updated', (data) => {
                this.handleContentUpdated(data);
            });
            
            this.ws.on('cursor_moved', (data) => {
                this.handleCursorMoved(data);
            });
        }
    }
    
    joinCollaboration() {
        if (!this.ws || !this.ws.isConnected()) {
            console.warn('WebSocket not connected, cannot join collaboration');
            return;
        }
        
        this.ws.emit('join_collaboration', {
            email_id: this.emailId
        });
        
        this.isActive = true;
        console.log('Joining collaboration for email:', this.emailId);
    }
    
    leaveCollaboration() {
        if (!this.ws || !this.isActive) {
            return;
        }
        
        this.ws.emit('leave_collaboration', {
            email_id: this.emailId
        });
        
        this.isActive = false;
        this.clearCollaborationUI();
        console.log('Left collaboration for email:', this.emailId);
    }
    
    pauseCollaboration() {
        if (this.isActive) {
            // Reduce update frequency when tab is not visible
            this.isActive = false;
        }
    }
    
    resumeCollaboration() {
        if (!this.isActive && this.emailId) {
            this.joinCollaboration();
        }
    }
    
    handleContentChange(event, type) {
        if (!this.isActive) return;
        
        const content = event.target.value;
        const timestamp = Date.now();
        
        // Store change in history
        this.contentHistory.push({
            type,
            content,
            timestamp,
            userId: this.getCurrentUserId()
        });
        
        // Limit history size
        if (this.contentHistory.length > 100) {
            this.contentHistory = this.contentHistory.slice(-50);
        }
        
        // Send update to other collaborators
        this.debouncedSendContentUpdate(content, type);
        
        // Show typing indicator
        this.showTypingIndicator();
    }
    
    handleCursorChange(event) {
        if (!this.isActive) return;
        
        const cursorPosition = event.target.selectionStart;
        this.debouncedSendCursorUpdate(cursorPosition);
    }
    
    sendContentUpdate(content, type = 'body') {
        if (!this.ws || !this.isActive) return;
        
        this.ws.emit('email_content_change', {
            email_id: this.emailId,
            content: content,
            content_type: type,
            timestamp: Date.now()
        });
    }
    
    sendCursorUpdate(position) {
        if (!this.ws || !this.isActive) return;
        
        this.ws.emit('cursor_update', {
            email_id: this.emailId,
            cursor_position: position
        });
    }
    
    handleCollaborationJoined(data) {
        console.log('Successfully joined collaboration:', data);
        
        if (data.active_users) {
            this.updateCollaborators(data.active_users);
        }
        
        this.showCollaborationStatus('Collaboration active', 'success');
    }
    
    handleUserJoined(data) {
        console.log('User joined:', data);
        
        if (data.user_id !== this.getCurrentUserId()) {
            this.addCollaborator(data);
            this.showCollaborationStatus(`${data.user_name} joined`, 'info');
        }
        
        if (data.active_users) {
            this.updateCollaborators(data.active_users);
        }
    }
    
    handleUserLeft(data) {
        console.log('User left:', data);
        
        this.removeCollaborator(data.user_id);
        
        if (data.active_users) {
            this.updateCollaborators(data.active_users);
        }
    }
    
    handleContentUpdated(data) {
        // Avoid infinite loops
        if (data.user_id === this.getCurrentUserId()) {
            return;
        }
        
        console.log('Content updated by collaborator:', data);
        
        // Apply the update
        this.applyContentUpdate(data);
        
        // Show indicator
        this.showContentUpdateIndicator(data.user_id);
    }
    
    handleCursorMoved(data) {
        if (data.user_id === this.getCurrentUserId()) {
            return;
        }
        
        this.updateCollaboratorCursor(data.user_id, data.cursor_position);
    }
    
    applyContentUpdate(data) {
        const targetElement = data.content_type === 'subject' ? 
            document.getElementById('emailSubject') : 
            document.getElementById('emailBody');
        
        if (!targetElement) return;
        
        // Store current cursor position
        const currentCursor = targetElement.selectionStart;
        const currentContent = targetElement.value;
        
        // Apply the update
        targetElement.value = data.content;
        
        // Try to maintain cursor position intelligently
        const newCursorPosition = this.calculateNewCursorPosition(
            currentContent, 
            data.content, 
            currentCursor
        );
        
        targetElement.setSelectionRange(newCursorPosition, newCursorPosition);
        
        // Add to content history
        this.contentHistory.push({
            type: data.content_type,
            content: data.content,
            timestamp: data.timestamp,
            userId: data.user_id,
            applied: true
        });
    }
    
    calculateNewCursorPosition(oldContent, newContent, oldCursor) {
        // Simple heuristic to maintain cursor position after collaborative updates
        const beforeCursor = oldContent.substring(0, oldCursor);
        const afterCursor = oldContent.substring(oldCursor);
        
        const beforeIndex = newContent.indexOf(beforeCursor);
        if (beforeIndex !== -1) {
            return beforeIndex + beforeCursor.length;
        }
        
        // Fallback: try to find the after-cursor content
        const afterIndex = newContent.indexOf(afterCursor);
        if (afterIndex !== -1) {
            return afterIndex;
        }
        
        // Fallback: proportional position
        const proportion = oldCursor / oldContent.length;
        return Math.round(newContent.length * proportion);
    }
    
    addCollaborator(userData) {
        this.collaborators.set(userData.user_id, {
            id: userData.user_id,
            name: userData.user_name,
            image: userData.user_image,
            joinedAt: Date.now(),
            lastSeen: Date.now(),
            cursorPosition: 0
        });
    }
    
    removeCollaborator(userId) {
        this.collaborators.delete(userId);
        this.removeCollaboratorCursor(userId);
    }
    
    updateCollaborators(activeUsers) {
        // Clear existing collaborators
        this.collaborators.clear();
        
        // Add current collaborators (excluding self)
        const currentUserId = this.getCurrentUserId();
        activeUsers.forEach(user => {
            if (user.user_id !== currentUserId) {
                this.addCollaborator(user);
            }
        });
        
        // Update UI
        this.updateCollaborationIndicator();
    }
    
    updateCollaborationIndicator() {
        const indicator = document.getElementById('collaborationIndicator');
        const count = document.getElementById('collaboratorCount');
        
        if (indicator && count) {
            const collaboratorCount = this.collaborators.size;
            
            if (collaboratorCount > 0) {
                indicator.classList.remove('d-none');
                count.textContent = collaboratorCount;
                
                // Update tooltip with collaborator names
                const names = Array.from(this.collaborators.values())
                    .map(c => c.name)
                    .join(', ');
                
                indicator.setAttribute('title', `Collaborating with: ${names}`);
            } else {
                indicator.classList.add('d-none');
            }
        }
        
        // Update collaborators modal content
        this.updateCollaboratorsModal();
    }
    
    updateCollaboratorsModal() {
        const collaboratorsDiv = document.getElementById('activeCollaborators');
        if (!collaboratorsDiv) return;
        
        const collaboratorsList = Array.from(this.collaborators.values());
        
        if (collaboratorsList.length === 0) {
            collaboratorsDiv.innerHTML = '<p class="text-muted">No active collaborators</p>';
            return;
        }
        
        collaboratorsDiv.innerHTML = collaboratorsList.map(collaborator => `
            <div class="d-flex align-items-center mb-3" data-user-id="${collaborator.id}">
                ${collaborator.image ? 
                    `<img src="${collaborator.image}" alt="${collaborator.name}" class="rounded-circle me-3" width="32" height="32" style="object-fit: cover;">` :
                    `<div class="bg-primary bg-opacity-25 rounded-circle me-3 d-flex align-items-center justify-content-center" style="width: 32px; height: 32px;">
                        <i data-feather="user" style="width: 16px; height: 16px;"></i>
                    </div>`
                }
                <div class="flex-grow-1">
                    <div class="fw-medium">${collaborator.name}</div>
                    <small class="text-muted">Active now</small>
                </div>
                <div class="status-indicator bg-success rounded-circle" style="width: 8px; height: 8px;"></div>
            </div>
        `).join('');
        
        // Re-initialize feather icons
        if (window.feather) {
            feather.replace();
        }
    }
    
    updateCollaboratorCursor(userId, position) {
        const collaborator = this.collaborators.get(userId);
        if (!collaborator) return;
        
        collaborator.cursorPosition = position;
        collaborator.lastSeen = Date.now();
        
        // Update visual cursor indicator
        this.showCollaboratorCursor(userId, position, collaborator);
    }
    
    showCollaboratorCursor(userId, position, collaborator) {
        const emailBody = document.getElementById('emailBody');
        if (!emailBody) return;
        
        // Remove existing cursor
        this.removeCollaboratorCursor(userId);
        
        // Create cursor element
        const cursor = document.createElement('div');
        cursor.id = `collaborator-cursor-${userId}`;
        cursor.className = 'collaborator-cursor position-absolute';
        cursor.style.borderLeft = `2px solid ${this.getUserColor(userId)}`;
        cursor.style.height = '20px';
        cursor.style.zIndex = '10';
        cursor.setAttribute('data-user', collaborator.name);
        
        // Calculate position (simplified - in production you'd need more sophisticated positioning)
        const rect = emailBody.getBoundingClientRect();
        const lineHeight = parseInt(getComputedStyle(emailBody).lineHeight) || 20;
        const charWidth = 8; // Approximate character width
        
        const lines = emailBody.value.substring(0, position).split('\n');
        const line = lines.length - 1;
        const column = lines[lines.length - 1].length;
        
        cursor.style.left = `${column * charWidth}px`;
        cursor.style.top = `${line * lineHeight}px`;
        
        // Add to container
        const container = emailBody.parentElement;
        if (container.style.position !== 'relative') {
            container.style.position = 'relative';
        }
        container.appendChild(cursor);
        
        // Auto-remove after 5 seconds of inactivity
        setTimeout(() => {
            if (Date.now() - collaborator.lastSeen > 5000) {
                this.removeCollaboratorCursor(userId);
            }
        }, 5000);
    }
    
    removeCollaboratorCursor(userId) {
        const cursor = document.getElementById(`collaborator-cursor-${userId}`);
        if (cursor) {
            cursor.remove();
        }
    }
    
    showTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.classList.remove('d-none');
            
            clearTimeout(this.typingTimeout);
            this.typingTimeout = setTimeout(() => {
                indicator.classList.add('d-none');
            }, 2000);
        }
    }
    
    showContentUpdateIndicator(userId) {
        const collaborator = this.collaborators.get(userId);
        const userName = collaborator ? collaborator.name : 'Someone';
        
        if (window.appUtils && window.appUtils.showToast) {
            window.appUtils.showToast(`${userName} updated the content`, 'info', 2000);
        }
        
        // Flash border of updated element
        const emailBody = document.getElementById('emailBody');
        if (emailBody) {
            const originalBorder = emailBody.style.border;
            emailBody.style.border = `2px solid ${this.getUserColor(userId)}`;
            
            setTimeout(() => {
                emailBody.style.border = originalBorder;
            }, 1000);
        }
    }
    
    showCollaborationStatus(message, type) {
        if (window.appUtils && window.appUtils.showToast) {
            window.appUtils.showToast(message, type, 3000);
        }
    }
    
    clearCollaborationUI() {
        // Remove all cursor indicators
        document.querySelectorAll('[id^="collaborator-cursor-"]').forEach(cursor => {
            cursor.remove();
        });
        
        // Hide collaboration indicator
        const indicator = document.getElementById('collaborationIndicator');
        if (indicator) {
            indicator.classList.add('d-none');
        }
        
        // Clear collaborators
        this.collaborators.clear();
    }
    
    // Utility methods
    getCurrentUserId() {
        return window.currentUser?.id || 'anonymous';
    }
    
    getUserColor(userId) {
        const colors = [
            '#0d6efd', '#20c997', '#ffc107', '#dc3545', 
            '#6f42c1', '#fd7e14', '#e91e63', '#00bcd4'
        ];
        const hash = userId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
        return colors[hash % colors.length];
    }
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    // History and undo functionality
    getContentHistory() {
        return this.contentHistory;
    }
    
    canUndo() {
        return this.contentHistory.length > 1;
    }
    
    undo() {
        if (!this.canUndo()) return false;
        
        // Get previous state
        this.contentHistory.pop(); // Remove current state
        const previousState = this.contentHistory[this.contentHistory.length - 1];
        
        if (previousState) {
            const targetElement = previousState.type === 'subject' ? 
                document.getElementById('emailSubject') : 
                document.getElementById('emailBody');
            
            if (targetElement) {
                targetElement.value = previousState.content;
                this.showCollaborationStatus('Undid last change', 'info');
                return true;
            }
        }
        
        return false;
    }
    
    // Public API
    isCollaborationActive() {
        return this.isActive && this.collaborators.size > 0;
    }
    
    getActiveCollaborators() {
        return Array.from(this.collaborators.values());
    }
    
    forceSync() {
        if (!this.isActive) return;
        
        const emailBody = document.getElementById('emailBody');
        const emailSubject = document.getElementById('emailSubject');
        
        if (emailBody) {
            this.sendContentUpdate(emailBody.value, 'body');
        }
        
        if (emailSubject) {
            this.sendContentUpdate(emailSubject.value, 'subject');
        }
    }
}

// Initialize collaboration when needed
let collaborationManager = null;

function initializeCollaboration(emailId) {
    if (!emailId || !window.socket) {
        console.warn('Cannot initialize collaboration: missing emailId or websocket');
        return null;
    }
    
    if (collaborationManager) {
        collaborationManager.leaveCollaboration();
    }
    
    collaborationManager = new CollaborationManager(emailId, window.socket);
    
    // Make available globally
    window.collaboration = collaborationManager;
    
    return collaborationManager;
}

// Export for use in other modules
window.CollaborationManager = CollaborationManager;
window.initializeCollaboration = initializeCollaboration;
