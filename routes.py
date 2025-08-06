from flask import render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import current_user
from app import app, db
from local_auth import require_login, local_auth
from models import (User, Team, TeamMember, TeamInvitation, Email, EmailTemplate, EmailAnalytics,
                   EmailStatus, UserRole, AIModel, EmailTone)
from ai_service import ai_service
from email_service import email_service
import json
import logging
from datetime import datetime, timedelta
import uuid

# Make datetime available to all templates
@app.context_processor
def inject_datetime():
    return dict(datetime=datetime)

# Register Local Auth blueprint
app.register_blueprint(local_auth)

# Make session permanent
@app.before_request
def make_session_permanent():
    session.permanent = True

@app.route('/')
def index():
    """Landing page - shows different content based on auth status"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@require_login
def dashboard():
    """Main dashboard for authenticated users"""
    try:
        # Get user's teams
        team_memberships = TeamMember.query.filter_by(user_id=current_user.id).all()
        teams = [membership.team for membership in team_memberships]

        # Get pending invitations
        pending_invitations = TeamInvitation.query.filter_by(
            invited_user_id=current_user.id,
            status='pending'
        ).all()

        # Get recent emails
        recent_emails = Email.query.filter_by(user_id=current_user.id)\
                                 .order_by(Email.created_at.desc())\
                                 .limit(10).all()

        # Get email analytics for the last 30 days
        start_date = datetime.now() - timedelta(days=30)
        analytics = email_service.get_email_analytics(
            user_id=current_user.id,
            start_date=start_date
        )

        return render_template('dashboard.html',
                             user=current_user,
                             teams=teams,
                             pending_invitations=pending_invitations,
                             recent_emails=recent_emails,
                             analytics=analytics.get('analytics', {}))
    except Exception as e:
        logging.error(f"Error loading dashboard: {str(e)}")
        flash('Error loading dashboard', 'error')
        return render_template('dashboard.html', user=current_user)

@app.route('/compose')
@require_login
def compose():
    """Email composition page"""
    try:
        # Get user's teams and templates
        team_memberships = TeamMember.query.filter_by(user_id=current_user.id).all()
        teams = [membership.team for membership in team_memberships]

        # Get templates (user's own + team templates)
        user_templates = EmailTemplate.query.filter_by(user_id=current_user.id).all()
        team_templates = []
        for team in teams:
            team_templates.extend(
                EmailTemplate.query.filter_by(team_id=team.id, is_public=True).all()
            )

        all_templates = user_templates + team_templates

        # Check if editing existing draft or loading an email to reply/forward
        email_id_param = request.args.get('edit') or request.args.get('email_id')

        draft_data = None
        if email_id_param:
            email = db.session.query(Email).filter_by(
                id=email_id_param,
                user_id=current_user.id
            ).first()

            if email:
                # Parse recipient addresses (handle both JSON objects and JSON strings)
                def parse_addresses_compose(addr_field):
                    if not addr_field:
                        return []
                    # If it's already a list (new JSON format), return as is
                    if isinstance(addr_field, list):
                        return addr_field
                    # If it's a string (old JSON string format), parse it
                    if isinstance(addr_field, str):
                        try:
                            parsed = json.loads(addr_field)
                            # Handle double-encoded JSON strings recursively
                            if isinstance(parsed, str):
                                try:
                                    parsed = json.loads(parsed)
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            return parsed if isinstance(parsed, list) else []
                        except (json.JSONDecodeError, TypeError):
                            return []
                    return []

                to_addresses = parse_addresses_compose(email.to_addresses)
                cc_addresses = parse_addresses_compose(email.cc_addresses)
                bcc_addresses = parse_addresses_compose(email.bcc_addresses)

                draft_data = {
                    'id': str(email.id),  # Ensure string format
                    'subject': email.subject or '',
                    'body_html': email.body_html or '',
                    'body_text': email.body_text or '',
                    'to_addresses': to_addresses,
                    'cc_addresses': cc_addresses,
                    'bcc_addresses': bcc_addresses,
                    'team_id': email.team_id,
                    'created_at': email.created_at.isoformat() if email.created_at else None,
                    'updated_at': email.updated_at.isoformat() if email.updated_at else None,
                    'status': email.status.value if email.status else 'draft',
                    'ai_model_used': email.ai_model_used.value if email.ai_model_used else None,
                    'tone_used': email.tone_used.value if email.tone_used else None
                }

        return render_template('compose.html',
                             user=current_user,
                             teams=teams,
                             templates=all_templates,
                             ai_models=list(AIModel),
                             email_tones=list(EmailTone),
                             draft_data=draft_data)
    except Exception as e:
        logging.error(f"Error loading compose page: {str(e)}")
        flash('Error loading compose page', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/generate-reply', methods=['POST'])
@require_login
def generate_reply():
    """API endpoint to generate AI email reply"""
    try:
        data = request.get_json()

        original_email = data.get('original_email', '')
        context = data.get('context', '')
        tone = data.get('tone', 'professional')
        model = data.get('model', 'auto')
        custom_instructions = data.get('custom_instructions', '')

        if not original_email:
            return jsonify({'success': False, 'error': 'Original email is required'}), 400

        # Generate AI reply
        result = ai_service.generate_email_reply(
            original_email=original_email,
            context=context,
            tone=tone,
            model=model,
            custom_instructions=custom_instructions
        )

        # Store the model used and generation time in the session for later use when saving draft
        if result.get('success'):
            session['last_ai_model_used'] = result.get('model_used')
            session['last_generation_time_ms'] = result.get('generation_time_ms', 2500)

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error generating reply: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save-draft', methods=['POST'])
@require_login
def save_draft():
    """Save email draft"""
    try:
        data = request.get_json()

        email_id = data.get('email_id')
        subject = data.get('subject', '')
        body_html = data.get('body_html', '')
        body_text = data.get('body_text', '')
        to_addresses = data.get('to_addresses', [])
        cc_addresses = data.get('cc_addresses', [])
        bcc_addresses = data.get('bcc_addresses', [])
        team_id = data.get('team_id')
        # Convert empty string to None for proper foreign key handling
        if team_id == '' or team_id is None:
            team_id = None

        if email_id:
            # Update existing draft
            email = Email.query.get(email_id)
            if not email or email.user_id != current_user.id:
                return jsonify({'success': False, 'error': 'Email not found or access denied'}), 404
        else:
            # Create new draft
            email = Email(
                id=str(uuid.uuid4()),
                user_id=current_user.id,
                team_id=team_id,
                status=EmailStatus.DRAFT
            )
            db.session.add(email)

        # Update email fields
        email.subject = subject
        email.body_html = body_html
        email.body_text = body_text
        email.to_addresses = to_addresses  # Store as JSON object
        email.cc_addresses = cc_addresses  # Store as JSON object
        email.bcc_addresses = bcc_addresses  # Store as JSON object
        email.team_id = team_id  # Update team_id as well
        email.updated_at = datetime.now()

        # Store AI model and generation time if it was used
        if 'last_ai_model_used' in session:
            model_string = session['last_ai_model_used']
            # Convert string to enum value
            if model_string == 'qwen-4-turbo':
                email.ai_model_used = AIModel.QWEN_4_TURBO
            elif model_string == 'claude-4-sonnet':
                email.ai_model_used = AIModel.CLAUDE_4_SONNET
            elif model_string == 'gpt-4o':
                email.ai_model_used = AIModel.GPT_4O

            # Store generation time if available
            if 'last_generation_time_ms' in session:
                email.generation_time_ms = session['last_generation_time_ms']
            elif email.generation_time_ms is None:
                # Set default generation time for AI-generated emails
                email.generation_time_ms = 2500

        db.session.commit()

        return jsonify({
            'success': True,
            'email_id': email.id,
            'message': 'Draft saved successfully'
        })

    except Exception as e:
        logging.error(f"Error saving draft: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/send-email', methods=['POST'])
@require_login
def send_email():
    """Send email"""
    try:
        data = request.get_json()

        email_id = data.get('email_id')
        if not email_id:
            return jsonify({'success': False, 'error': 'Email ID is required'}), 400

        email = Email.query.get(email_id)
        if not email or email.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'Email not found or access denied'}), 404

        # Parse recipient addresses (handle both JSON objects and JSON strings)
        def parse_addresses(addr_field):
            if not addr_field:
                return []
            # If it's already a list (new JSON format), return as is
            if isinstance(addr_field, list):
                return addr_field
            # If it's a string (old JSON string format), parse it
            if isinstance(addr_field, str):
                try:
                    parsed = json.loads(addr_field)
                    # Handle double-encoded JSON strings recursively
                    if isinstance(parsed, str):
                        try:
                            parsed = json.loads(parsed)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    return parsed if isinstance(parsed, list) else []
                except (json.JSONDecodeError, TypeError):
                    return []
            return []

        to_addresses = parse_addresses(email.to_addresses)
        cc_addresses = parse_addresses(email.cc_addresses)
        bcc_addresses = parse_addresses(email.bcc_addresses)
        
        # Debug logging
        logging.info(f"Parsed addresses - To: {to_addresses} (type: {type(to_addresses)})")
        logging.info(f"Parsed addresses - CC: {cc_addresses} (type: {type(cc_addresses)})")
        logging.info(f"Parsed addresses - BCC: {bcc_addresses} (type: {type(bcc_addresses)})")

        # Validate required fields
        if not to_addresses or not email.subject:
            return jsonify({'success': False, 'error': 'To addresses and subject are required'}), 400

        # Send email
        result = email_service.send_email(
            user=current_user,
            to_addresses=to_addresses,
            subject=email.subject,
            body_html=email.body_html,
            body_text=email.body_text,
            cc_addresses=cc_addresses,
            bcc_addresses=bcc_addresses,
            email_id=email.id
        )

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/templates')
@require_login
def templates():
    """Email templates management page"""
    try:
        # Get user's teams with error handling
        team_memberships = []
        try:
            team_memberships = TeamMember.query.filter_by(user_id=current_user.id).all()
        except Exception as e:
            logging.warning(f"Error loading team memberships: {str(e)}")

        teams = [membership.team for membership in team_memberships if membership.team]

        # Get user's templates with error handling
        user_templates = []
        try:
            user_templates = EmailTemplate.query.filter_by(user_id=current_user.id).all()
        except Exception as e:
            logging.warning(f"Error loading user templates: {str(e)}")

        # Get team templates (if user is admin/manager) with error handling
        team_templates = []
        try:
            for membership in team_memberships:
                if membership.role in [UserRole.ADMIN, UserRole.MANAGER]:
                    team_templates.extend(
                        EmailTemplate.query.filter_by(team_id=membership.team_id).all()
                    )
        except Exception as e:
            logging.warning(f"Error loading team templates: {str(e)}")

        return render_template('templates.html',
                             user=current_user,
                             teams=teams or [],
                             user_templates=user_templates or [],
                             team_templates=team_templates or [],
                             email_tones=list(EmailTone))
    except Exception as e:
        logging.error(f"Critical error loading templates: {str(e)}")
        return render_template('500.html'), 500

@app.route('/api/save-template', methods=['POST'])
@require_login
def save_template():
    """Save email template"""
    try:
        data = request.get_json()

        template_id = data.get('template_id')
        name = data.get('name', '')
        description = data.get('description', '')
        subject_template = data.get('subject_template', '')
        body_template = data.get('body_template', '')
        default_tone = data.get('default_tone')
        is_public = data.get('is_public', False)
        team_id = data.get('team_id')

        if not name or not body_template:
            return jsonify({'success': False, 'error': 'Name and body template are required'}), 400

        if template_id:
            # Update existing template
            template = EmailTemplate.query.get(template_id)
            if not template or template.user_id != current_user.id:
                return jsonify({'success': False, 'error': 'Template not found or access denied'}), 404
        else:
            # Create new template
            template = EmailTemplate(
                id=str(uuid.uuid4()),
                user_id=current_user.id,
                team_id=team_id
            )
            db.session.add(template)

        # Update template fields
        template.name = name
        template.description = description
        template.subject_template = subject_template
        template.body_template = body_template
        template.default_tone = EmailTone(default_tone) if default_tone else None
        template.is_public = is_public
        template.updated_at = datetime.now()

        db.session.commit()

        return jsonify({
            'success': True,
            'template_id': template.id,
            'message': 'Template saved successfully'
        })

    except Exception as e:
        logging.error(f"Error saving template: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/team')
@require_login
def team():
    """Team management page"""
    try:
        # Get user's teams
        team_memberships = TeamMember.query.filter_by(user_id=current_user.id).all()
        teams = []

        for membership in team_memberships:
            # Get pending invitations for teams where user can manage
            pending_invitations = []
            if membership.role in [UserRole.ADMIN, UserRole.MANAGER]:
                pending_invitations = TeamInvitation.query.filter_by(
                    team_id=membership.team_id,
                    status='pending'
                ).all()
            
            team_data = {
                'team': membership.team,
                'role': membership.role,
                'members': membership.team.members,
                'pending_invitations': pending_invitations
            }
            teams.append(team_data)

        return render_template('team.html',
                             user=current_user,
                             teams=teams,
                             user_roles=list(UserRole))
    except Exception as e:
        logging.error(f"Error loading team page: {str(e)}")
        flash('Error loading team page', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/create-team', methods=['POST'])
@require_login
def create_team():
    """Create a new team"""
    try:
        data = request.get_json()

        name = data.get('name', '').strip()
        description = data.get('description', '')

        if not name:
            return jsonify({'success': False, 'error': 'Team name is required'}), 400

        # Create team
        team = Team(
            id=str(uuid.uuid4()),
            name=name,
            description=description
        )
        db.session.add(team)

        # Add creator as admin
        membership = TeamMember(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            team_id=team.id,
            role=UserRole.ADMIN
        )
        db.session.add(membership)

        db.session.commit()

        return jsonify({
            'success': True,
            'team_id': team.id,
            'message': 'Team created successfully'
        })

    except Exception as e:
        logging.error(f"Error creating team: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/invite-member', methods=['POST'])
@require_login
def invite_member():
    """Send invitation to a team member"""
    try:
        data = request.get_json()
        
        team_id = data.get('team_id')
        email = data.get('email', '').strip()
        role = data.get('role', 'user')
        message = data.get('message', '')
        
        if not all([team_id, email]):
            return jsonify({'success': False, 'error': 'Team ID and email are required'}), 400
        
        # Check if user has permission to invite (must be admin or manager)
        user_membership = TeamMember.query.filter_by(
            user_id=current_user.id,
            team_id=team_id
        ).first()
        
        if not user_membership or user_membership.role not in [UserRole.ADMIN, UserRole.MANAGER]:
            return jsonify({'success': False, 'error': 'You do not have permission to invite members'}), 403
        
        # Check if team exists
        team = Team.query.get(team_id)
        if not team:
            return jsonify({'success': False, 'error': 'Team not found'}), 404
        
        # Check if user exists
        invited_user = User.query.filter_by(email=email).first()
        if not invited_user:
            return jsonify({'success': False, 'error': 'User with this email not found. They need to register first.'}), 404
        
        # Check if user is already a member
        existing_membership = TeamMember.query.filter_by(
            user_id=invited_user.id,
            team_id=team_id
        ).first()
        
        if existing_membership:
            return jsonify({'success': False, 'error': 'User is already a member of this team'}), 400
        
        # Check if invitation already exists
        existing_invitation = TeamInvitation.query.filter_by(
            team_id=team_id,
            invited_user_id=invited_user.id,
            status='pending'
        ).first()
        
        if existing_invitation:
            return jsonify({'success': False, 'error': 'Invitation already sent to this user'}), 400
        
        # Create invitation
        invitation = TeamInvitation(
            id=str(uuid.uuid4()),
            team_id=team_id,
            invited_user_id=invited_user.id,
            invited_by_id=current_user.id,
            role=UserRole(role),
            message=message
        )
        db.session.add(invitation)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Invitation sent to {invited_user.first_name or email}'
        })
        
    except Exception as e:
        logging.error(f"Error sending invitation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/change-member-role', methods=['POST'])
@require_login
def change_member_role():
    """Change a team member's role"""
    try:
        data = request.get_json()
        
        member_id = data.get('member_id')
        new_role = data.get('role')
        
        if not all([member_id, new_role]):
            return jsonify({'success': False, 'error': 'Member ID and role are required'}), 400
        
        # Get the membership to change
        membership = TeamMember.query.get(member_id)
        if not membership:
            return jsonify({'success': False, 'error': 'Team member not found'}), 404
        
        # Check if current user has permission (must be admin)
        user_membership = TeamMember.query.filter_by(
            user_id=current_user.id,
            team_id=membership.team_id
        ).first()
        
        if not user_membership or user_membership.role != UserRole.ADMIN:
            return jsonify({'success': False, 'error': 'Only team admins can change member roles'}), 403
        
        # Don't allow changing own role if you're the only admin
        if membership.user_id == current_user.id:
            admin_count = TeamMember.query.filter_by(
                team_id=membership.team_id,
                role=UserRole.ADMIN
            ).count()
            
            if admin_count == 1 and new_role != 'admin':
                return jsonify({'success': False, 'error': 'Cannot change role - you are the only admin'}), 400
        
        # Update role
        membership.role = UserRole(new_role)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Member role updated successfully'
        })
        
    except Exception as e:
        logging.error(f"Error changing member role: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/remove-member', methods=['DELETE'])
@require_login
def remove_member():
    """Remove a team member"""
    try:
        data = request.get_json()
        member_id = data.get('member_id')
        
        if not member_id:
            return jsonify({'success': False, 'error': 'Member ID is required'}), 400
        
        # Get the membership to remove
        membership = TeamMember.query.get(member_id)
        if not membership:
            return jsonify({'success': False, 'error': 'Team member not found'}), 404
        
        # Check if current user has permission (must be admin)
        user_membership = TeamMember.query.filter_by(
            user_id=current_user.id,
            team_id=membership.team_id
        ).first()
        
        if not user_membership or user_membership.role != UserRole.ADMIN:
            return jsonify({'success': False, 'error': 'Only team admins can remove members'}), 403
        
        # Don't allow removing yourself if you're the only admin
        if membership.user_id == current_user.id:
            admin_count = TeamMember.query.filter_by(
                team_id=membership.team_id,
                role=UserRole.ADMIN
            ).count()
            
            if admin_count == 1:
                return jsonify({'success': False, 'error': 'Cannot remove yourself - you are the only admin'}), 400
        
        # Remove membership
        db.session.delete(membership)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Member removed successfully'
        })
        
    except Exception as e:
        logging.error(f"Error removing member: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-invitations')
@require_login
def get_invitations():
    """Get pending invitations for current user"""
    try:
        # Get pending invitations
        invitations = TeamInvitation.query.filter_by(
            invited_user_id=current_user.id,
            status='pending'
        ).all()
        
        invitation_data = []
        for invitation in invitations:
            invitation_data.append({
                'id': invitation.id,
                'team': {
                    'id': invitation.team.id,
                    'name': invitation.team.name,
                    'description': invitation.team.description
                },
                'invited_by': {
                    'name': invitation.invited_by.first_name or invitation.invited_by.email.split('@')[0],
                    'email': invitation.invited_by.email
                },
                'role': invitation.role.value,
                'message': invitation.message,
                'created_at': invitation.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'invitations': invitation_data
        })
        
    except Exception as e:
        logging.error(f"Error getting invitations: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/respond-invitation', methods=['POST'])
@require_login
def respond_invitation():
    """Accept or decline team invitation"""
    try:
        data = request.get_json()
        
        invitation_id = data.get('invitation_id')
        response = data.get('response')  # 'accept' or 'decline'
        
        if not all([invitation_id, response]):
            return jsonify({'success': False, 'error': 'Invitation ID and response are required'}), 400
        
        if response not in ['accept', 'decline']:
            return jsonify({'success': False, 'error': 'Response must be "accept" or "decline"'}), 400
        
        # Get the invitation
        invitation = TeamInvitation.query.get(invitation_id)
        if not invitation:
            return jsonify({'success': False, 'error': 'Invitation not found'}), 404
        
        # Check if invitation belongs to current user
        if invitation.invited_user_id != current_user.id:
            return jsonify({'success': False, 'error': 'This invitation does not belong to you'}), 403
        
        # Check if invitation is still pending
        if invitation.status != 'pending':
            return jsonify({'success': False, 'error': 'This invitation has already been responded to'}), 400
        
        # Update invitation status
        invitation.status = 'accepted' if response == 'accept' else 'declined'
        invitation.responded_at = datetime.now()
        
        message = ''
        
        if response == 'accept':
            # Check if user is already a member (race condition protection)
            existing_membership = TeamMember.query.filter_by(
                user_id=current_user.id,
                team_id=invitation.team_id
            ).first()
            
            if not existing_membership:
                # Create team membership
                membership = TeamMember(
                    id=str(uuid.uuid4()),
                    user_id=current_user.id,
                    team_id=invitation.team_id,
                    role=invitation.role
                )
                db.session.add(membership)
                message = f'Successfully joined {invitation.team.name}'
            else:
                message = f'You are already a member of {invitation.team.name}'
        else:
            message = f'Declined invitation to {invitation.team.name}'
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': message
        })
        
    except Exception as e:
        logging.error(f"Error responding to invitation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cancel-invitation', methods=['DELETE'])
@require_login
def cancel_invitation():
    """Cancel a pending invitation"""
    try:
        data = request.get_json()
        invitation_id = data.get('invitation_id')
        
        if not invitation_id:
            return jsonify({'success': False, 'error': 'Invitation ID is required'}), 400
        
        # Get the invitation
        invitation = TeamInvitation.query.get(invitation_id)
        if not invitation:
            return jsonify({'success': False, 'error': 'Invitation not found'}), 404
        
        # Check if current user has permission (must be admin/manager of the team or the person who sent the invitation)
        user_membership = TeamMember.query.filter_by(
            user_id=current_user.id,
            team_id=invitation.team_id
        ).first()
        
        if not (invitation.invited_by_id == current_user.id or 
                (user_membership and user_membership.role in [UserRole.ADMIN, UserRole.MANAGER])):
            return jsonify({'success': False, 'error': 'You do not have permission to cancel this invitation'}), 403
        
        # Check if invitation is still pending
        if invitation.status != 'pending':
            return jsonify({'success': False, 'error': 'Can only cancel pending invitations'}), 400
        
        # Delete the invitation
        db.session.delete(invitation)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Invitation cancelled successfully'
        })
        
    except Exception as e:
        logging.error(f"Error cancelling invitation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/analytics')
@require_login
def analytics():
    """Analytics dashboard"""
    try:
        # Get user's teams for filtering
        team_memberships = TeamMember.query.filter_by(user_id=current_user.id).all()
        teams = [membership.team for membership in team_memberships]

        # Get analytics for last 30 days
        start_date = datetime.now() - timedelta(days=30)

        # User analytics
        user_analytics = email_service.get_email_analytics(
            user_id=current_user.id,
            start_date=start_date
        )

        # Team analytics (if user has access)
        team_analytics = []
        for membership in team_memberships:
            if membership.role in [UserRole.ADMIN, UserRole.MANAGER]:
                team_data = email_service.get_email_analytics(
                    team_id=membership.team_id,
                    start_date=start_date
                )
                if team_data['success']:
                    team_analytics.append({
                        'team': membership.team,
                        'analytics': team_data['analytics']
                    })

        return render_template('analytics.html',
                             user=current_user,
                             teams=teams,
                             user_analytics=user_analytics.get('analytics', {}),
                             team_analytics=team_analytics)
    except Exception as e:
        logging.error(f"Error loading analytics: {str(e)}")
        flash('Error loading analytics', 'error')
        return redirect(url_for('dashboard'))

@app.route('/docs')
def api_docs():
    """Show comprehensive API documentation with FastAPI integration"""
    return render_template('api_documentation.html')

@app.route('/api/docs')
def api_docs_alt():
    """Serve FastAPI documentation through Flask proxy"""
    try:
        import requests
        response = requests.get('http://localhost:8000/docs', timeout=10)
        if response.status_code == 200:
            # Replace localhost:8000 with the current host for proper API calls
            content = response.text.replace('http://localhost:8000', '/fastapi-proxy')
            return content, 200, {'Content-Type': 'text/html'}
        else:
            return jsonify({'error': 'FastAPI docs not available', 'status': response.status_code}), 503
    except Exception as e:
        return jsonify({'error': f'Cannot reach FastAPI service: {str(e)}'}), 503

@app.route('/docs/fastapi')
def fastapi_docs_direct():
    """Direct link to FastAPI documentation"""
    return redirect('http://localhost:8000/docs')

@app.route('/fastapi-proxy/<path:path>')
def fastapi_proxy(path):
    """Proxy requests to FastAPI service"""
    try:
        import requests
        url = f'http://localhost:8000/{path}'

        if request.method == 'GET':
            response = requests.get(url, params=request.args, timeout=10)
        elif request.method == 'POST':
            response = requests.post(url, json=request.get_json(), timeout=10)
        else:
            response = requests.request(request.method, url, timeout=10)

        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI proxy error: {str(e)}'}), 503

@app.route('/openapi.json')
def openapi_json():
    """Proxy OpenAPI JSON from FastAPI service"""
    try:
        import requests
        response = requests.get('http://localhost:8000/openapi.json', timeout=10)
        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI OpenAPI error: {str(e)}'}), 503

# FastAPI endpoint proxies
@app.route('/api/v1/models')
def get_models():
    """Proxy to FastAPI models endpoint"""
    try:
        import requests
        response = requests.get('http://localhost:8000/api/v1/models', timeout=10)
        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI models error: {str(e)}'}), 503

@app.route('/api/v1/generate-email', methods=['POST'])
def generate_email():
    """Proxy to FastAPI email generation endpoint"""
    try:
        import requests
        response = requests.post(
            'http://localhost:8000/api/v1/generate-email',
            json=request.get_json(),
            timeout=30
        )
        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI generate-email error: {str(e)}'}), 503

@app.route('/api/v1/analyze-email', methods=['POST'])
def analyze_email():
    """Proxy to FastAPI email analysis endpoint"""
    try:
        import requests
        response = requests.post(
            'http://localhost:8000/api/v1/analyze-email',
            json=request.get_json(),
            timeout=30
        )
        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI analyze-email error: {str(e)}'}), 503

@app.route('/api/v1/bulk-generate', methods=['POST'])
def bulk_generate():
    """Proxy to FastAPI bulk generation endpoint"""
    try:
        import requests
        response = requests.post(
            'http://localhost:8000/api/v1/bulk-generate',
            json=request.get_json(),
            timeout=60
        )
        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI bulk-generate error: {str(e)}'}), 503

@app.route('/api/v1/health')
def health_check():
    """Proxy to FastAPI health check endpoint"""
    try:
        import requests
        response = requests.get('http://localhost:8000/api/v1/health', timeout=10)
        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI health error: {str(e)}'}), 503

@app.route('/api/v1/generate-template', methods=['POST'])
def generate_template_fastapi():
    """Proxy to FastAPI template generation endpoint"""
    try:
        import requests
        response = requests.post(
            'http://localhost:8000/api/v1/generate-template',
            json=request.get_json(),
            timeout=30
        )
        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI generate-template error: {str(e)}'}), 503

@app.route('/api/v1/langchain-query', methods=['POST'])
def langchain_query():
    """Proxy to FastAPI LangChain agent query endpoint"""
    try:
        import requests
        response = requests.post(
            'http://localhost:8000/api/v1/langchain-query',
            json=request.get_json(),
            timeout=60
        )
        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI langchain-query error: {str(e)}'}), 503

@app.route('/api/v1/enhanced-generate', methods=['POST'])
def enhanced_generate():
    """Proxy to FastAPI enhanced email generation endpoint"""
    try:
        import requests
        response = requests.post(
            'http://localhost:8000/api/v1/enhanced-generate',
            json=request.get_json(),
            timeout=60
        )
        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI enhanced-generate error: {str(e)}'}), 503

@app.route('/api/v1/enhanced-analyze', methods=['POST'])
def enhanced_analyze():
    """Proxy to FastAPI enhanced email analysis endpoint"""
    try:
        import requests
        response = requests.post(
            'http://localhost:8000/api/v1/enhanced-analyze',
            json=request.get_json(),
            timeout=60
        )
        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI enhanced-analyze error: {str(e)}'}), 503

@app.route('/api/v1/langchain-status')
def langchain_status():
    """Proxy to FastAPI LangChain status endpoint"""
    try:
        import requests
        response = requests.get('http://localhost:8000/api/v1/langchain-status', timeout=10)
        return response.content, response.status_code, dict(response.headers)
    except Exception as e:
        return jsonify({'error': f'FastAPI langchain-status error: {str(e)}'}), 503

@app.route('/fastapi')
def fastapi_root():
    """Redirect to FastAPI root endpoint"""
    return redirect('/fastapi-proxy/')

@app.route('/api/v1/status')
def api_status():
    """Get status of both Flask and FastAPI services"""
    try:
        import requests
        fastapi_status = requests.get('http://localhost:8000/api/v1/health', timeout=5)
        fastapi_data = fastapi_status.json() if fastapi_status.status_code == 200 else {"status": "error"}
    except Exception as e:
        fastapi_data = {"status": "error", "error": str(e)}

    return jsonify({
        "flask_service": {
            "status": "operational",
            "port": 5000,
            "features": ["frontend", "authentication", "database", "websockets"],
            "endpoints": ["/", "/dashboard", "/compose", "/docs", "/api/generate-reply"]
        },
        "fastapi_service": fastapi_data,
        "hybrid_architecture": {
            "description": "Flask frontend + FastAPI backend with LangChain integration",
            "flask_port": 5000,
            "fastapi_port": 8000,
            "langchain_enabled": True,
            "openrouter_qwen_available": True
        },
        "documentation": {
            "comprehensive_docs": "/docs",
            "fastapi_interactive": "http://localhost:8000/docs",
            "service_status": "/api/v1/status"
        }
    })

@app.route('/settings')
@require_login
def settings():
    """User settings page"""
    try:
        # Get common SMTP providers for dropdown
        smtp_providers = {
            'gmail': email_service.get_common_smtp_settings('gmail'),
            'outlook': email_service.get_common_smtp_settings('outlook'),
            'yahoo': email_service.get_common_smtp_settings('yahoo'),
            'custom': email_service.get_common_smtp_settings('custom')
        }

        return render_template('settings.html',
                             user=current_user,
                             smtp_providers=smtp_providers,
                             ai_models=list(AIModel),
                             email_tones=list(EmailTone))
    except Exception as e:
        logging.error(f"Error loading settings: {str(e)}")
        flash('Error loading settings', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/update-smtp-settings', methods=['POST'])
@require_login
def update_smtp_settings():
    """Update user's SMTP settings"""
    try:
        data = request.get_json()

        smtp_server = data.get('smtp_server', '').strip()
        smtp_port = data.get('smtp_port')
        smtp_username = data.get('smtp_username', '').strip()
        smtp_password = data.get('smtp_password', '').strip()
        smtp_use_tls = data.get('smtp_use_tls', True)

        if not all([smtp_server, smtp_port, smtp_username, smtp_password]):
            return jsonify({'success': False, 'error': 'All SMTP fields are required'}), 400

        # Test connection first
        test_result = email_service.test_smtp_connection(
            smtp_server=smtp_server,
            smtp_port=int(smtp_port),
            smtp_username=smtp_username,
            smtp_password=smtp_password,
            use_tls=smtp_use_tls
        )

        if not test_result['success']:
            return jsonify(test_result), 400

        # Update user settings
        current_user.smtp_server = smtp_server
        current_user.smtp_port = int(smtp_port)
        current_user.smtp_username = smtp_username
        current_user.smtp_password = smtp_password  # Note: Should encrypt this in production
        current_user.smtp_use_tls = smtp_use_tls

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'SMTP settings updated successfully'
        })

    except Exception as e:
        logging.error(f"Error updating SMTP settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/test-smtp-connection', methods=['POST'])
@require_login
def test_smtp_connection():
    """Test SMTP connection with provided settings"""
    try:
        data = request.get_json()

        smtp_server = data.get('smtp_server', '').strip()
        smtp_port = data.get('smtp_port')
        smtp_username = data.get('smtp_username', '').strip()
        smtp_password = data.get('smtp_password', '').strip()
        smtp_use_tls = data.get('smtp_use_tls', True)

        if not all([smtp_server, smtp_port, smtp_username, smtp_password]):
            return jsonify({'success': False, 'error': 'All SMTP fields are required'}), 400

        # Test connection
        result = email_service.test_smtp_connection(
            smtp_server=smtp_server,
            smtp_port=int(smtp_port),
            smtp_username=smtp_username,
            smtp_password=smtp_password,
            use_tls=smtp_use_tls
        )

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error testing SMTP connection: {str(e)}")
        return jsonify({'success': False, 'error': f'Connection test failed: {str(e)}'}), 500

@app.route('/api/analyze-sentiment', methods=['POST'])
@require_login
def analyze_sentiment():
    """Analyze email sentiment using AI"""
    try:
        data = request.get_json()
        email_content = data.get('email_content', '')

        if not email_content:
            return jsonify({'success': False, 'error': 'Email content is required'}), 400

        result = ai_service.analyze_email_with_langchain(email_content)
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error analyzing sentiment: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/suggest-improvements', methods=['POST'])
@require_login
def suggest_improvements():
    """Get AI suggestions for email improvements"""
    try:
        data = request.get_json()
        email_content = data.get('email_content', '')

        if not email_content:
            return jsonify({'success': False, 'error': 'Email content is required'}), 400

        result = ai_service.suggest_email_improvements(email_content)
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error getting suggestions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-email/<email_id>')
@require_login
def get_email(email_id):
    """Get email details by ID"""
    try:
        email = Email.query.get(email_id)
        if not email or email.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'Email not found or access denied'}), 404

        email_data = {
            'id': email.id,
            'subject': email.subject,
            'body_html': email.body_html,
            'body_text': email.body_text,
            'to_addresses': email.to_addresses or [],
            'cc_addresses': email.cc_addresses or [],
            'bcc_addresses': email.bcc_addresses or [],
            'status': email.status.value if email.status else 'unknown',
            'ai_model_used': email.ai_model_used.value if email.ai_model_used else None,
            'tone_used': email.tone_used.value if email.tone_used else None,
            'original_email': email.original_email,
            'context': email.context,
            'created_at': email.created_at.isoformat() if email.created_at else None,
            'updated_at': email.updated_at.isoformat() if email.updated_at else None,
            'sent_at': email.sent_at.isoformat() if email.sent_at else None,
            'delivered_at': email.delivered_at.isoformat() if email.delivered_at else None,
            'opened_at': email.opened_at.isoformat() if email.opened_at else None,
            'replied_at': email.replied_at.isoformat() if email.replied_at else None,
            'user_rating': email.user_rating,
            'generation_time_ms': email.generation_time_ms
        }

        return jsonify({
            'success': True,
            'email': email_data
        })

    except Exception as e:
        logging.error(f"Error getting email details: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/load-draft/<email_id>')
@require_login
def load_draft(email_id):
    """Load draft content for editing"""
    try:
        email = Email.query.get(email_id)
        if not email or email.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'Draft not found or access denied'}), 404

        draft_data = {
            'id': email.id,
            'subject': email.subject or '',
            'body_html': email.body_html or '',
            'body_text': email.body_text or '',
            'to_addresses': email.to_addresses or [],
            'cc_addresses': email.cc_addresses or [],
            'bcc_addresses': email.bcc_addresses or [],
            'team_id': email.team_id,
            'status': email.status.value if email.status else 'draft',
            'ai_model_used': email.ai_model_used.value if email.ai_model_used else None,
            'tone_used': email.tone_used.value if email.tone_used else None,
            'created_at': email.created_at.isoformat() if email.created_at else None,
            'updated_at': email.updated_at.isoformat() if email.updated_at else None
        }

        return jsonify({
            'success': True,
            'draft': draft_data
        })

    except Exception as e:
        logging.error(f"Error loading draft: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete-template', methods=['DELETE'])
@require_login
def delete_template():
    """Delete email template"""
    try:
        data = request.get_json()
        template_id = data.get('template_id')

        if not template_id:
            return jsonify({'success': False, 'error': 'Template ID is required'}), 400

        template = EmailTemplate.query.get(template_id)
        if not template or template.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'Template not found or access denied'}), 404

        db.session.delete(template)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Template deleted successfully'
        })

    except Exception as e:
        logging.error(f"Error deleting template: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-template', methods=['POST'])
@require_login
def generate_template():
    """Generate email template using AI"""
    try:
        data = request.get_json()

        template_type = data.get('template_type', 'professional')
        purpose = data.get('purpose', '')
        tone = data.get('tone', 'professional')
        industry = data.get('industry', '')
        custom_instructions = data.get('custom_instructions', '')

        if not purpose:
            return jsonify({'success': False, 'error': 'Template purpose is required'}), 400

        result = ai_service.generate_email_template(
            template_type=template_type,
            purpose=purpose,
            tone=tone,
            industry=industry,
            custom_instructions=custom_instructions
        )

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error generating template: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete-draft/<email_id>', methods=['DELETE'])
@require_login
def delete_draft(email_id):
    """Delete email draft"""
    try:
        email = Email.query.get(email_id)
        if not email or email.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'Draft not found or access denied'}), 404

        if email.status != EmailStatus.DRAFT:
            return jsonify({'success': False, 'error': 'Only drafts can be deleted'}), 400

        db.session.delete(email)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Draft deleted successfully'
        })

    except Exception as e:
        logging.error(f"Error deleting draft: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500