from flask_socketio import emit, join_room, leave_room, rooms
from flask_login import current_user
from app import socketio, db
from models import CollaborationSession, Email, EmailDraft
import json
import logging
from datetime import datetime

# Store active collaboration sessions
active_sessions = {}

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    if current_user.is_authenticated:
        logging.info(f"User {current_user.id} connected to WebSocket")
        emit('connected', {'message': 'Connected to AI Email Assistant'})
    else:
        logging.warning("Unauthenticated user attempted WebSocket connection")
        return False

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    if current_user.is_authenticated:
        logging.info(f"User {current_user.id} disconnected from WebSocket")
        # Clean up any active collaboration sessions
        cleanup_user_sessions(current_user.id)

@socketio.on('join_collaboration')
def handle_join_collaboration(data):
    """Join a collaborative editing session for an email"""
    if not current_user.is_authenticated:
        return
    
    email_id = data.get('email_id')
    if not email_id:
        emit('error', {'message': 'Email ID is required'})
        return
    
    # Verify user has access to this email
    email = Email.query.get(email_id)
    if not email:
        emit('error', {'message': 'Email not found'})
        return
    
    # Check if user has permission to edit this email
    if email.user_id != current_user.id and not has_team_access(current_user.id, email.team_id):
        emit('error', {'message': 'Permission denied'})
        return
    
    # Join the collaboration room
    room_name = f"email_{email_id}"
    join_room(room_name)
    
    # Create or update collaboration session
    session = CollaborationSession.query.filter_by(
        email_id=email_id,
        user_id=current_user.id
    ).first()
    
    if not session:
        session = CollaborationSession(
            email_id=email_id,
            user_id=current_user.id,
            is_active=True
        )
        db.session.add(session)
    else:
        session.is_active = True
        session.last_seen = datetime.now()
    
    db.session.commit()
    
    # Store session info
    if email_id not in active_sessions:
        active_sessions[email_id] = {}
    
    active_sessions[email_id][current_user.id] = {
        'user_name': f"{current_user.first_name} {current_user.last_name}".strip() or current_user.email,
        'user_image': current_user.profile_image_url,
        'cursor_position': 0,
        'last_seen': datetime.now().isoformat()
    }
    
    # Notify other users in the room
    emit('user_joined', {
        'user_id': current_user.id,
        'user_name': active_sessions[email_id][current_user.id]['user_name'],
        'user_image': current_user.profile_image_url,
        'active_users': list(active_sessions[email_id].values())
    }, room=room_name)
    
    # Send current active users to the joining user
    emit('collaboration_joined', {
        'email_id': email_id,
        'active_users': list(active_sessions[email_id].values())
    })

@socketio.on('leave_collaboration')
def handle_leave_collaboration(data):
    """Leave a collaborative editing session"""
    if not current_user.is_authenticated:
        return
    
    email_id = data.get('email_id')
    if not email_id:
        return
    
    room_name = f"email_{email_id}"
    leave_room(room_name)
    
    # Update collaboration session
    session = CollaborationSession.query.filter_by(
        email_id=email_id,
        user_id=current_user.id
    ).first()
    
    if session:
        session.is_active = False
        db.session.commit()
    
    # Remove from active sessions
    if email_id in active_sessions and current_user.id in active_sessions[email_id]:
        del active_sessions[email_id][current_user.id]
        
        # Clean up empty sessions
        if not active_sessions[email_id]:
            del active_sessions[email_id]
    
    # Notify other users
    emit('user_left', {
        'user_id': current_user.id,
        'active_users': list(active_sessions.get(email_id, {}).values())
    }, room=room_name)

@socketio.on('email_content_change')
def handle_email_content_change(data):
    """Handle real-time email content changes"""
    if not current_user.is_authenticated:
        return
    
    email_id = data.get('email_id')
    content = data.get('content', '')
    cursor_position = data.get('cursor_position', 0)
    
    if not email_id:
        return
    
    # Update cursor position in active sessions
    if email_id in active_sessions and current_user.id in active_sessions[email_id]:
        active_sessions[email_id][current_user.id]['cursor_position'] = cursor_position
        active_sessions[email_id][current_user.id]['last_seen'] = datetime.now().isoformat()
    
    # Save draft version
    try:
        draft = EmailDraft.query.filter_by(
            email_id=email_id,
            user_id=current_user.id,
            is_active=True
        ).first()
        
        if not draft:
            draft = EmailDraft(
                email_id=email_id,
                user_id=current_user.id,
                content=content,
                version=1
            )
            db.session.add(draft)
        else:
            draft.content = content
            draft.version += 1
        
        db.session.commit()
        
        # Broadcast changes to other users in the room
        room_name = f"email_{email_id}"
        emit('content_updated', {
            'user_id': current_user.id,
            'content': content,
            'cursor_position': cursor_position,
            'timestamp': datetime.now().isoformat(),
            'version': draft.version
        }, room=room_name, include_self=False)
        
    except Exception as e:
        logging.error(f"Error saving email draft: {str(e)}")
        emit('error', {'message': 'Failed to save changes'})

@socketio.on('cursor_update')
def handle_cursor_update(data):
    """Handle cursor position updates"""
    if not current_user.is_authenticated:
        return
    
    email_id = data.get('email_id')
    cursor_position = data.get('cursor_position', 0)
    
    if not email_id:
        return
    
    # Update cursor position
    if email_id in active_sessions and current_user.id in active_sessions[email_id]:
        active_sessions[email_id][current_user.id]['cursor_position'] = cursor_position
        active_sessions[email_id][current_user.id]['last_seen'] = datetime.now().isoformat()
    
    # Broadcast cursor position to other users
    room_name = f"email_{email_id}"
    emit('cursor_moved', {
        'user_id': current_user.id,
        'cursor_position': cursor_position
    }, room=room_name, include_self=False)

@socketio.on('ai_generation_start')
def handle_ai_generation_start(data):
    """Notify users that AI generation has started"""
    if not current_user.is_authenticated:
        return
    
    email_id = data.get('email_id')
    model = data.get('model', 'auto')
    
    if not email_id:
        return
    
    room_name = f"email_{email_id}"
    emit('ai_generation_started', {
        'user_id': current_user.id,
        'model': model,
        'timestamp': datetime.now().isoformat()
    }, room=room_name)

@socketio.on('ai_generation_complete')
def handle_ai_generation_complete(data):
    """Notify users that AI generation is complete"""
    if not current_user.is_authenticated:
        return
    
    email_id = data.get('email_id')
    content = data.get('content', '')
    model_used = data.get('model_used')
    generation_time = data.get('generation_time_ms', 0)
    
    if not email_id:
        return
    
    room_name = f"email_{email_id}"
    emit('ai_generation_completed', {
        'user_id': current_user.id,
        'content': content,
        'model_used': model_used,
        'generation_time_ms': generation_time,
        'timestamp': datetime.now().isoformat()
    }, room=room_name)

@socketio.on('email_sent')
def handle_email_sent(data):
    """Notify team members when an email is sent"""
    if not current_user.is_authenticated:
        return
    
    email_id = data.get('email_id')
    recipients = data.get('recipients', [])
    
    if not email_id:
        return
    
    # Get email details
    email = Email.query.get(email_id)
    if not email:
        return
    
    # Notify team members if this is a team email
    if email.team_id:
        room_name = f"team_{email.team_id}"
        emit('email_sent_notification', {
            'email_id': email_id,
            'subject': email.subject,
            'sender': f"{current_user.first_name} {current_user.last_name}".strip() or current_user.email,
            'recipients': recipients,
            'timestamp': datetime.now().isoformat()
        }, room=room_name)

def cleanup_user_sessions(user_id: str):
    """Clean up collaboration sessions for a disconnected user"""
    try:
        # Mark database sessions as inactive
        sessions = CollaborationSession.query.filter_by(
            user_id=user_id,
            is_active=True
        ).all()
        
        for session in sessions:
            session.is_active = False
        
        db.session.commit()
        
        # Remove from active sessions memory
        sessions_to_clean = []
        for email_id, users in active_sessions.items():
            if user_id in users:
                del users[user_id]
                sessions_to_clean.append((email_id, users))
        
        # Notify remaining users in affected sessions
        for email_id, remaining_users in sessions_to_clean:
            room_name = f"email_{email_id}"
            socketio.emit('user_left', {
                'user_id': user_id,
                'active_users': list(remaining_users.values())
            }, room=room_name)
        
    except Exception as e:
        logging.error(f"Error cleaning up user sessions: {str(e)}")

def has_team_access(user_id: str, team_id: str) -> bool:
    """Check if user has access to a team"""
    if not team_id:
        return False
    
    from models import TeamMember
    member = TeamMember.query.filter_by(
        user_id=user_id,
        team_id=team_id
    ).first()
    
    return member is not None

# Periodic cleanup of stale sessions
@socketio.on('ping')
def handle_ping():
    """Handle ping for keeping connection alive"""
    if current_user.is_authenticated:
        emit('pong')
