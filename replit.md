# AI Email Assistant

## Overview

The AI Email Assistant is a Flask-based web application designed to revolutionize professional email communication through AI-powered assistance. The application leverages multiple AI models (Qwen-4, Claude-4, and GPT-4o) to generate contextually appropriate email replies, provide intelligent suggestions, and enable real-time team collaboration on email drafts. Built natively for Replit's cloud infrastructure, it offers enterprise-grade features including multi-tenant support, advanced analytics, and zero-configuration deployment.

## User Preferences

Preferred communication style: Simple, everyday language.
AI Model Integration: Successfully added OpenRouter Qwen model (qwen/qwen3-30b-a3b-instruct-2507) for professional and technical email generation.
LangChain Enhancement: Comprehensive LangChain functionality integrated with chains, memory, agents, and structured output parsing.
Date: August 5, 2025

## Recent Changes

### August 6, 2025
- ✓ Created complete local authentication system replacing Replit auth
- ✓ Built professional login and registration pages with form validation
- ✓ Added password_hash column to database for secure credential storage
- ✓ Fixed all template URL references to use local authentication
- ✓ Resolved profile dropdown route conflicts causing unexpected errors
- ✓ Application fully functional with local user management
- ✓ Enhanced sentiment analysis with advanced LLM-powered LangChain implementation
- ✓ Replaced basic keyword suggestions with AI-powered improvement recommendations
- ✓ Fixed frontend-backend API response structure compatibility for suggestions feature
- ✓ Added comprehensive scoring metrics (clarity, professionalism, engagement)
- ✓ Fixed critical email sending error: 'str' object has no attribute 'extend'
- ✓ Resolved recipient email address display issue (no longer shows comma-separated characters)
- ✓ Updated JSON storage format for email addresses for consistency
- ✓ Added robust parsing to handle both old and new email address formats
- ✓ **SPECTACULAR AI-POWERED TEMPLATE GENERATOR**: Created advanced LangChain-based system
- ✓ Multi-model AI orchestration with intelligent model selection (Qwen-4, Claude-4, GPT-4o)
- ✓ Advanced prompt engineering with sophisticated template generation capabilities
- ✓ Smart placeholder generation and industry-specific customization features
- ✓ Dynamic complexity assessment and intelligent categorization system
- ✓ Fixed LangChain prompt template escaping issues for proper JSON generation
- ✓ Successfully tested template generation with OpenRouter Qwen model integration

### August 5, 2025
- ✓ Successfully started AI Email Assistant with OpenRouter API integration
- ✓ Fixed missing dependencies (uvicorn, fastapi) and database connection
- ✓ Application running on dual ports: Flask frontend (5000) + FastAPI backend (8000)
- ✓ OpenRouter API key configured for Qwen model access
- ✓ Created simple Hello World program as requested
- ✓ Diagnosed compose email visibility issue: authentication required for API endpoints

## Current Status
- **Hello World Program**: ✓ Working correctly, prints "Hello World"
- **AI Email Assistant**: ✓ Running successfully with local authentication system
- **Local Authentication**: ✓ Complete login/registration system with secure password storage
- **Profile Management**: ✓ Settings and logout functionality working properly
- **OpenRouter API**: ✓ Fixed authentication headers for OpenRouter API integration

## System Architecture

### Frontend Architecture
- **Technology Stack**: Bootstrap 5.3 with custom CSS, Feather Icons for UI consistency
- **Real-time Features**: Socket.IO integration for live collaboration and instant updates
- **Client-side Components**: Modular JavaScript architecture with dedicated managers for WebSocket, collaboration, and application state
- **Theme Support**: Built-in light/dark theme switching with localStorage persistence
- **Responsive Design**: Mobile-first approach with progressive enhancement

### Backend Architecture
- **Framework**: Flask with SQLAlchemy ORM for database operations
- **Database Layer**: PostgreSQL with declarative base models using SQLAlchemy
- **Authentication**: Replit Auth integration with OAuth2 flow and Flask-Login session management
- **API Design**: RESTful endpoints with JSON responses and proper HTTP status codes
- **Real-time Communication**: Flask-SocketIO for WebSocket connections supporting team collaboration

### AI Integration Layer
- **Comprehensive LangChain Integration**: Full LangChain ecosystem with chains, memory, agents, and output parsers
- **Enhanced AI Service**: Advanced AI orchestration with conversational agents and structured output parsing
- **LangChain Chains**: Sequential email processing, analysis, and template generation chains
- **Conversation Memory**: Buffer and summary memory systems for context-aware interactions
- **AI Agents**: Conversational ReAct agents with specialized email processing tools
- **FastAPI Backend**: High-performance async API layer running on port 8000 for advanced AI operations
- **Multi-Model Support**: Dynamic AI model selection based on context (Qwen-4 Turbo, Claude-4 Sonnet, GPT-4o)
- **Provider Abstraction**: Unified interface supporting OpenAI, Anthropic, and OpenRouter APIs through LangChain
- **Context-Aware Prompting**: Specialized system prompts with LangChain ChatPromptTemplate for different email types
- **Hybrid Architecture**: Flask frontend (port 5000) + FastAPI backend (port 8000) for optimal performance
- **Cost Optimization**: Token usage tracking and intelligent model routing based on complexity
- **Structured Output**: Pydantic models for email responses, analysis, and template structures

### Data Models
- **User Management**: Complete user profiles with SMTP configuration and preferences
- **Team Collaboration**: Multi-tenant team structure with role-based permissions (Admin, Manager, User, Viewer)
- **Email Operations**: Full email lifecycle tracking with status management and analytics
- **Template System**: Reusable email templates with team sharing capabilities

### Security Architecture
- **Authentication**: OAuth2 with Replit Auth, session-based user management
- **Data Protection**: Encrypted SMTP credentials storage, secure environment variable handling
- **Authorization**: Role-based access control for team resources and email operations
- **Input Validation**: Comprehensive form validation and sanitization

## External Dependencies

### AI Service Providers
- **OpenAI API**: GPT-4o model integration for concise and urgent email responses
- **Anthropic API**: Claude-4 Sonnet for creative and analytical email generation
- **OpenRouter**: Qwen-4 Turbo access for professional and technical communications

### Replit Platform Services
- **Replit Database**: PostgreSQL managed instance for primary data storage
- **Replit Auth**: OAuth2 authentication service for user management
- **Replit Secrets**: Secure environment variable management for API keys and credentials

### Frontend Libraries
- **Bootstrap 5.3**: UI framework for responsive design and components
- **Socket.IO**: Real-time bidirectional communication for collaboration features
- **Chart.js**: Data visualization for email analytics and performance metrics
- **Feather Icons**: Consistent iconography throughout the application

### Python Dependencies
- **Flask Ecosystem**: Core framework with SQLAlchemy, SocketIO, and Login extensions
- **LangChain Suite**: LangChain core, OpenAI, Anthropic, and Community integrations for AI orchestration
- **FastAPI Framework**: High-performance async API framework with Pydantic validation
- **AI Libraries**: OpenAI and Anthropic SDKs for model integration
- **Email Services**: SMTP libraries for email sending capabilities
- **Security**: JWT and OAuth libraries for authentication handling

### SMTP Integration
- **Email Sending**: User-configurable SMTP servers for email delivery
- **Connection Pooling**: Efficient SMTP connection management
- **Multi-provider Support**: Compatible with Gmail, Outlook, and custom SMTP servers