from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import logging
from ai_service import AIService
from email_service import EmailService
import asyncio
from datetime import datetime

# Initialize FastAPI app
fastapi_app = FastAPI(
    title="AI Email Assistant API",
    description="FastAPI layer for AI-powered email generation with LangChain integration",
    version="1.0.0"
)

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ai_service = AIService()
email_service = EmailService()

# LangChain integration is built into the main ai_service
ENHANCED_AI_AVAILABLE = True
logging.info("LangChain functionality available through main AI service")

# Pydantic models for request/response
class EmailGenerationRequest(BaseModel):
    original_email: str = Field(..., description="The original email to reply to")
    context: Optional[str] = Field("", description="Additional context for the reply")
    tone: Optional[str] = Field("professional", description="Tone of the reply")
    model: Optional[str] = Field("auto", description="AI model to use (auto, qwen-4-turbo, claude-4-sonnet, gpt-4o)")
    custom_instructions: Optional[str] = Field("", description="Custom instructions for the AI")
    user_id: Optional[int] = Field(None, description="User ID for analytics")

class EmailGenerationResponse(BaseModel):
    success: bool
    email_reply: Optional[str] = None
    error: Optional[str] = None
    model_used: str
    generation_time_ms: int
    tokens_used: Optional[int] = None
    method: str = "langchain"

class EmailAnalysisRequest(BaseModel):
    email_content: str = Field(..., description="Email content to analyze")

class EmailAnalysisResponse(BaseModel):
    sentiment: str
    urgency: str
    tone: str
    emotion_score: float
    key_topics: List[str]

class BulkEmailRequest(BaseModel):
    emails: List[EmailGenerationRequest] = Field(..., description="List of emails to process")
    parallel: Optional[bool] = Field(True, description="Process emails in parallel")

class ModelStatusResponse(BaseModel):
    available_models: List[str]
    langchain_models: List[str]
    default_model: str
    model_details: Dict[str, Any]

class TemplateGenerationRequest(BaseModel):
    template_type: Optional[str] = Field("professional", description="Type of template to generate")
    purpose: str = Field(..., description="Purpose of the email template")
    tone: Optional[str] = Field("professional", description="Tone of the template")
    industry: Optional[str] = Field("", description="Industry context")
    custom_instructions: Optional[str] = Field("", description="Custom instructions for generation")

class TemplateGenerationResponse(BaseModel):
    success: bool
    template_name: Optional[str] = None
    subject_template: Optional[str] = None
    body_template: Optional[str] = None
    error: Optional[str] = None
    model_used: str
    generation_time_ms: int

class LangChainRequest(BaseModel):
    query: str = Field(..., description="Query for the LangChain agent")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context tracking")
    use_memory: Optional[bool] = Field(True, description="Whether to use conversation memory")

class LangChainResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    conversation_id: Optional[str] = None
    error: Optional[str] = None
    model_used: str
    chains_used: List[str] = []
    memory_used: bool = False

# API Endpoints
@fastapi_app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Email Assistant FastAPI Service",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "features": ["email_generation", "langchain_integration", "sentiment_analysis"]
    }

@fastapi_app.get("/api/v1/models", response_model=ModelStatusResponse)
async def get_available_models():
    """Get information about available AI models"""
    try:
        langchain_models = list(ai_service.langchain_models.keys())
        from ai_service import AI_MODELS
        
        return ModelStatusResponse(
            available_models=list(AI_MODELS.keys()),
            langchain_models=langchain_models,
            default_model="qwen-4-turbo",
            model_details=AI_MODELS
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model information: {str(e)}")

@fastapi_app.post("/api/v1/generate-email", response_model=EmailGenerationResponse)
async def generate_email_reply(request: EmailGenerationRequest):
    """Generate an AI-powered email reply using LangChain"""
    try:
        result = ai_service.generate_email_reply(
            original_email=request.original_email,
            context=request.context,
            tone=request.tone,
            model=request.model,
            custom_instructions=request.custom_instructions
        )
        
        return EmailGenerationResponse(**result)
        
    except Exception as e:
        logging.error(f"Error generating email reply: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating email: {str(e)}")

@fastapi_app.post("/api/v1/analyze-email", response_model=EmailAnalysisResponse)
async def analyze_email(request: EmailAnalysisRequest):
    """Analyze email sentiment and characteristics"""
    try:
        result = ai_service.analyze_email_sentiment(request.email_content)
        return EmailAnalysisResponse(**result)
        
    except Exception as e:
        logging.error(f"Error analyzing email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing email: {str(e)}")

@fastapi_app.post("/api/v1/bulk-generate")
async def bulk_generate_emails(request: BulkEmailRequest, background_tasks: BackgroundTasks):
    """Generate multiple email replies in parallel"""
    try:
        if request.parallel:
            # Process emails in parallel using asyncio
            tasks = []
            for email_req in request.emails:
                task = asyncio.create_task(
                    asyncio.to_thread(
                        ai_service.generate_email_reply,
                        email_req.original_email,
                        email_req.context,
                        email_req.tone,
                        email_req.model,
                        email_req.custom_instructions
                    )
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            responses = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    responses.append({
                        "success": False,
                        "error": str(result),
                        "email_index": i,
                        "model_used": request.emails[i].model,
                        "generation_time_ms": 0,
                        "method": "bulk_parallel"
                    })
                else:
                    result["email_index"] = i
                    result["method"] = "bulk_parallel"
                    responses.append(result)
                    
            return {"results": responses, "total_processed": len(responses)}
        else:
            # Process emails sequentially
            responses = []
            for i, email_req in enumerate(request.emails):
                try:
                    result = ai_service.generate_email_reply(
                        email_req.original_email,
                        email_req.context,
                        email_req.tone,
                        email_req.model,
                        email_req.custom_instructions
                    )
                    result["email_index"] = i
                    result["method"] = "bulk_sequential"
                    responses.append(result)
                except Exception as e:
                    responses.append({
                        "success": False,
                        "error": str(e),
                        "email_index": i,
                        "model_used": email_req.model,
                        "generation_time_ms": 0,
                        "method": "bulk_sequential"
                    })
            
            return {"results": responses, "total_processed": len(responses)}
            
    except Exception as e:
        logging.error(f"Error in bulk email generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in bulk generation: {str(e)}")

@fastapi_app.get("/api/v1/health")
async def health_check():
    """Detailed health check including model availability"""
    try:
        model_status = {}
        for model_name in ai_service.langchain_models:
            try:
                # Test model with a simple prompt
                test_result = ai_service.generate_email_reply(
                    original_email="Test email",
                    context="Quick health check",
                    model=model_name,
                    tone="professional"
                )
                model_status[model_name] = "operational" if test_result.get("success") else "error"
            except Exception:
                model_status[model_name] = "unavailable"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": model_status,
            "features": {
                "langchain_integration": True,
                "openrouter_qwen": "qwen-4-turbo" in ai_service.langchain_models,
                "anthropic_claude": "claude-4-sonnet" in ai_service.langchain_models,
                "openai_gpt": "gpt-4o" in ai_service.langchain_models
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@fastapi_app.post("/api/v1/generate-template", response_model=TemplateGenerationResponse)
async def generate_email_template(request: TemplateGenerationRequest):
    """Generate an AI-powered email template using LangChain"""
    try:
        start_time = datetime.now()
        
        result = ai_service.generate_email_template(
            template_type=request.template_type,
            purpose=request.purpose,
            tone=request.tone,
            industry=request.industry,
            custom_instructions=request.custom_instructions
        )
        
        end_time = datetime.now()
        generation_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        if result.get('success'):
            return TemplateGenerationResponse(
                success=True,
                template_name=result.get('template_name'),
                subject_template=result.get('subject_template'),
                body_template=result.get('body_template'),
                model_used=result.get('model_used', 'qwen-4-turbo'),
                generation_time_ms=generation_time_ms
            )
        else:
            return TemplateGenerationResponse(
                success=False,
                error=result.get('error', 'Unknown error'),
                model_used=result.get('model_used', 'qwen-4-turbo'),
                generation_time_ms=generation_time_ms
            )
        
    except Exception as e:
        logging.error(f"Error generating email template: {str(e)}")
        return TemplateGenerationResponse(
            success=False,
            error=str(e),
            model_used="qwen-4-turbo",
            generation_time_ms=0
        )

@fastapi_app.post("/api/v1/langchain-query", response_model=LangChainResponse)
async def langchain_query(request: LangChainRequest):
    """Process complex email tasks using LangChain agents and chains"""
    try:
        start_time = datetime.now()
        
        if not ENHANCED_AI_AVAILABLE:
            return LangChainResponse(
                success=False,
                error="Enhanced LangChain service not available",
                model_used="none",
                chains_used=[],
                memory_used=False
            )
        
        # Use LangChain conversational agent for advanced query processing
        result = ai_service.process_with_conversational_agent(
            query=request.query,
            conversation_id=request.conversation_id
        )
        
        end_time = datetime.now()
        
        if result.get('success'):
            return LangChainResponse(
                success=True,
                response=result.get('agent_response'),
                conversation_id=result.get('conversation_id'),
                model_used=result.get('model_used', 'qwen-4-turbo'),
                chains_used=['conversational_agent'],
                memory_used=request.use_memory
            )
        else:
            return LangChainResponse(
                success=False,
                error=result.get('error', 'Unknown error'),
                model_used=result.get('model_used', 'qwen-4-turbo'),
                chains_used=[],
                memory_used=False
            )
        
    except Exception as e:
        logging.error(f"Error in LangChain query: {str(e)}")
        return LangChainResponse(
            success=False,
            error=str(e),
            model_used="qwen-4-turbo",
            chains_used=[],
            memory_used=False
        )

@fastapi_app.post("/api/v1/enhanced-generate", response_model=EmailGenerationResponse)
async def enhanced_email_generation(request: EmailGenerationRequest):
    """Enhanced email generation using LangChain chains and memory"""
    try:
        start_time = datetime.now()
        
        # Use comprehensive LangChain email generation
        result = ai_service.generate_email_reply_with_langchain(
            original_email=request.original_email,
            context=request.context,
            tone=request.tone,
            custom_instructions=request.custom_instructions
        )
        
        end_time = datetime.now()
        generation_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        if result.get('success'):
            return EmailGenerationResponse(
                success=True,
                email_reply=result.get('email_response') or result.get('reply'),
                model_used=result.get('model_used', 'qwen-4-turbo'),
                generation_time_ms=generation_time_ms,
                tokens_used=result.get('tokens_used'),
                method="enhanced_langchain" if ENHANCED_AI_AVAILABLE else "standard"
            )
        else:
            return EmailGenerationResponse(
                success=False,
                error=result.get('error', 'Unknown error'),
                model_used=result.get('model_used', 'qwen-4-turbo'),
                generation_time_ms=generation_time_ms,
                method="enhanced_langchain" if ENHANCED_AI_AVAILABLE else "standard"
            )
        
    except Exception as e:
        logging.error(f"Error in enhanced email generation: {str(e)}")
        return EmailGenerationResponse(
            success=False,
            error=str(e),
            model_used="qwen-4-turbo",
            generation_time_ms=0,
            method="error"
        )

@fastapi_app.post("/api/v1/enhanced-analyze", response_model=EmailAnalysisResponse)
async def enhanced_email_analysis(request: EmailAnalysisRequest):
    """Enhanced email analysis using LangChain chains"""
    try:
        # Use comprehensive LangChain email analysis
        result = ai_service.analyze_email_with_langchain(request.email_content)
        
        return EmailAnalysisResponse(
            sentiment=result.get('sentiment', 'neutral'),
            urgency=result.get('urgency', 'medium'),
            tone=result.get('tone', 'professional'),
            emotion_score=result.get('emotion_score', 0.5),
            key_topics=result.get('key_topics', []),
            method="standard"
        )
        
    except Exception as e:
        logging.error(f"Error in enhanced email analysis: {str(e)}")
        return EmailAnalysisResponse(
            sentiment="neutral",
            urgency="medium",
            tone="professional",
            emotion_score=0.5,
            key_topics=[],
            method="error"
        )

@fastapi_app.get("/api/v1/langchain-status")
async def langchain_status():
    """Get comprehensive LangChain integration status"""
    try:
        return {
            "langchain_available": True,
            "enhanced_service": "integrated_in_main_service",
            "available_models": list(ai_service.langchain_models.keys()) if hasattr(ai_service, 'langchain_models') else [],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "langchain_available": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Error handlers
@fastapi_app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "Please check the API documentation"}

@fastapi_app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "An unexpected error occurred"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)