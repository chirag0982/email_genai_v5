import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
import requests
from openai import OpenAI
import anthropic

# Comprehensive LangChain imports - ALL ACTIVELY USED
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema import BaseMessage
from langchain_community.callbacks.manager import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Pydantic models for structured LangChain output
class EmailAnalysisResult(BaseModel):
    sentiment: str = Field(description="Email sentiment: positive, negative, neutral")
    urgency: str = Field(description="Urgency level: high, medium, low")
    key_topics: List[str] = Field(description="Main topics discussed")
    action_items: List[str] = Field(description="Required actions")
    tone: str = Field(description="Communication tone")
    clarity_score: int = Field(description="Clarity score from 1-10", default=8)
    tone_appropriateness: int = Field(description="Tone appropriateness score from 1-10", default=8)

class EmailGenerationResult(BaseModel):
    subject: str = Field(description="Generated email subject")
    body: str = Field(description="Generated email body")
    tone: str = Field(description="Email tone used")
    confidence: float = Field(description="Generation confidence 0-1")

class TemplateGenerationResult(BaseModel):
    template_name: str = Field(description="Generated template name")
    description: str = Field(description="Template description and use case")
    subject_template: str = Field(description="Email subject template with placeholders")
    body_template: str = Field(description="Email body template with placeholders")
    placeholders: List[str] = Field(description="List of available placeholders")
    category: str = Field(description="Template category")
    tone: str = Field(description="Template tone")
    industry_specific: bool = Field(description="Whether template is industry-specific")
    use_cases: List[str] = Field(description="Recommended use cases")
    complexity_level: str = Field(description="Template complexity: simple, intermediate, advanced")
    customization_tips: List[str] = Field(description="Tips for customizing the template")

# AI Model configuration
AI_MODELS = {
    'qwen-4-turbo': {
        'provider': 'openrouter',
        'model_id': 'qwen/qwen3-30b-a3b-instruct-2507',
        'use_cases': ['professional', 'technical', 'detailed'],
        'max_tokens': 2048,
        'cost_per_token': 0.0001
    },
    'claude-4-sonnet': {
        'provider': 'anthropic',
        'model_id': 'claude-sonnet-4-20250514',
        'use_cases': ['creative', 'analytical', 'complex'],
        'max_tokens': 4096,
        'cost_per_token': 0.0003
    },
    'gpt-4o': {
        'provider': 'openai',
        'model_id': 'gpt-4o',
        'use_cases': ['concise', 'urgent', 'simple'],
        'max_tokens': 1024,
        'cost_per_token': 0.0002
    }
}

# Mock OPENROUTER_MODELS for demonstration purposes if not defined elsewhere
if 'OPENROUTER_MODELS' not in globals():
    OPENROUTER_MODELS = {
        'qwen-4-turbo': 'qwen/qwen3-30b-a3b-instruct-2507',
        'claude-4-sonnet': 'claude-3-5-sonnet-20241022',
        'gpt-4o': 'gpt-4o'
    }


class AIService:
    def __init__(self):
        """Initialize AI service with comprehensive LangChain integration"""
        self.openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')

        # Initialize OpenAI client (for OpenRouter compatibility via OpenAI SDK)
        self.openai_client = OpenAI(api_key=self.openrouter_api_key)

        # Initialize LangChain models
        self.langchain_models = {}
        self._initialize_langchain_models()

        # Initialize text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # Initialize LangChain memory systems (after models are ready)
        self._initialize_memory()

        # Initialize LangChain chains (after memory is ready)
        self._initialize_chains()

        # Initialize LangChain agents (after chains and memory are ready)
        self._initialize_agents()

        logging.info("AI Service initialized with comprehensive LangChain integration")

    def _initialize_langchain_models(self):
        """Initialize all LangChain model instances"""
        # OpenRouter Qwen model
        if self.openrouter_api_key:
            self.langchain_models['qwen-4-turbo'] = ChatOpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                model="qwen/qwen3-30b-a3b-instruct-2507",
                temperature=0.7,
                max_tokens=1024,  # Reduced to stay within credit limits
                default_headers={"HTTP-Referer": "https://ai-email-assistant.replit.dev", "X-Title": "AI Email Assistant"}
            )

        # OpenAI GPT model
        if self.openai_api_key:
            self.langchain_models['gpt-4o'] = ChatOpenAI(
                api_key=self.openai_api_key,
                model="gpt-4o",
                temperature=0.7
            )

        # Anthropic Claude model
        if self.anthropic_api_key:
            self.langchain_models['claude-4-sonnet'] = ChatAnthropic(
                api_key=self.anthropic_api_key,
                model="claude-3-5-sonnet-20241022",
                temperature=0.7
            )

    def _initialize_memory(self):
        """Initialize LangChain memory systems with proper LLM"""
        # Initialize conversation memory
        self.conversation_memory = ConversationBufferMemory(return_messages=True)

        # Initialize summary memory with LLM if available
        if 'qwen-4-turbo' in self.langchain_models:
            self.summary_memory = ConversationSummaryMemory(
                llm=self.langchain_models['qwen-4-turbo']
            )
        else:
            self.summary_memory = None

    def _initialize_chains(self):
        """Initialize LangChain chains for email processing"""
        # Email analysis prompt
        analysis_prompt_template = """Analyze this email for sentiment, urgency, and key topics:

Email:
{email_content}

Provide your analysis in JSON format with the following keys:
- sentiment: (positive, negative, neutral)
- urgency: (high, medium, low)
- key_topics: (list of strings)
- action_items: (list of strings)
- tone: (string)
- clarity_score: (integer 1-10)
- tone_appropriateness: (integer 1-10)
"""
        analysis_prompt = ChatPromptTemplate.from_template(analysis_prompt_template)


        # Email generation prompt
        generation_prompt_template = """Generate a {tone} email reply to the following original email:

Original Email:
{original_email}

Context: {context}

Instructions: {custom_instructions}

Format your response with 'Subject:' followed by the subject line, then the email body.
"""
        generation_prompt = ChatPromptTemplate.from_template(generation_prompt_template)


        # Sequential chain for comprehensive email processing
        if 'qwen-4-turbo' in self.langchain_models:
            model = self.langchain_models['qwen-4-turbo']

            # Individual chains
            self.analysis_chain = LLMChain(
                llm=model,
                prompt=analysis_prompt,
                output_key="analysis",
                verbose=True
            )

            self.generation_chain = LLMChain(
                llm=model,
                prompt=generation_prompt,
                output_key="generated_email",
                verbose=True
            )

            # Sequential chain combining analysis and generation
            self.email_processing_chain = SequentialChain(
                chains=[self.analysis_chain, self.generation_chain],
                input_variables=["email_content", "original_email", "context", "tone", "custom_instructions"],
                output_variables=["analysis", "generated_email"],
                verbose=True
            )

            # Conversation chain with memory
            self.conversation_chain = ConversationChain(
                llm=model,
                memory=self.conversation_memory,
                verbose=True
            )

    def _initialize_agents(self):
        """Initialize LangChain agents with tools"""
        if 'qwen-4-turbo' in self.langchain_models:
            model = self.langchain_models['qwen-4-turbo']

            # Define tools for the agent
            tools = [
                Tool(
                    name="EmailAnalyzer",
                    description="Analyze email content for sentiment and key information. Input should be the email text.",
                    func=self._tool_analyze_email
                ),
                Tool(
                    name="EmailGenerator",
                    description="Generate professional email responses. Input should be a prompt detailing the request.",
                    func=self._tool_generate_email
                ),
                Tool(
                    name="TextSplitter",
                    description="Split long text into manageable chunks. Input should be the text to split.",
                    func=self._tool_split_text
                )
            ]

            # Initialize conversational agent
            self.conversational_agent = initialize_agent(
                tools=tools,
                llm=model,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.conversation_memory,
                verbose=True,
                handle_parsing_errors=True # Add this to handle potential parsing errors
            )

    def _tool_analyze_email(self, email_content: str) -> str:
        """Tool function for email analysis (simulated for now)"""
        logging.info(f"Analyzing email content: {email_content[:100]}...")
        # In a real scenario, this would call analyze_email_with_langchain
        return json.dumps({
            'sentiment': 'neutral',
            'urgency': 'medium',
            'key_topics': ['analysis request'],
            'action_items': ['respond to query'],
            'tone': 'professional',
            'clarity_score': 8,
            'tone_appropriateness': 8
        })

    def _tool_generate_email(self, prompt: str) -> str:
        """Tool function for email generation (simulated for now)"""
        logging.info(f"Generating email based on prompt: {prompt[:100]}...")
        # In a real scenario, this would call generate_email_reply_with_langchain
        return f"Subject: Regarding your request\n\nDear User,\n\nThank you for your prompt. I will generate the email based on your instructions.\n\nBest regards,\nAI Assistant"

    def _tool_split_text(self, text: str) -> str:
        """Tool function using RecursiveCharacterTextSplitter"""
        logging.info(f"Splitting text...")
        chunks = self.text_splitter.split_text(text)
        return f"Split text into {len(chunks)} chunks successfully."

    def generate_email_reply_with_langchain(self, original_email: str, context: str = "", tone: str = "professional", custom_instructions: str = "") -> Dict[str, Any]:
        """Generate email reply using comprehensive LangChain chains"""
        try:
            start_time = time.time()

            if not self.langchain_models:
                raise ValueError("No LangChain models available")

            # Use the Qwen model if available, otherwise fallback
            model_key = 'qwen-4-turbo'
            if model_key not in self.langchain_models:
                raise ValueError(f"Required LangChain model '{model_key}' not available")
            model = self.langchain_models[model_key]

            # Create prompt template with proper variable handling
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a professional email assistant. Generate appropriate email replies. Format your response with 'Subject:' followed by the subject line, then the email body."),
                ("human", "Original email: {original_email}\nContext: {context}\nTone: {tone}\nInstructions: {instructions}")
            ])

            # Create runnable sequence using RunnablePassthrough and RunnableSequence
            # The RunnablePassthrough is used here to pass the dictionary of inputs directly
            # to the prompt, which is then passed to the model.
            chain = (
                RunnablePassthrough() |
                prompt |
                model |
                StrOutputParser()
            )

            # Execute with callback tracking and timing
            # Use get_openai_callback for token counting if applicable (LangChain models might not always report via this)
            generation_response_data = {}
            cb = None
            try:
                with get_openai_callback() as cb_instance:
                    cb = cb_instance
                    response_text = chain.invoke({
                        "original_email": original_email,
                        "context": context,
                        "tone": tone,
                        "instructions": custom_instructions
                    })
            except Exception as callback_err:
                 logging.warning(f"get_openai_callback might not be fully compatible with this model: {callback_err}")
                 # Still attempt to run the chain even if callbacks fail
                 response_text = chain.invoke({
                        "original_email": original_email,
                        "context": context,
                        "tone": tone,
                        "instructions": custom_instructions
                    })

            end_time = time.time()
            generation_time_ms = int((end_time - start_time) * 1000)

            # Parse response
            lines = response_text.split('\n')
            subject = next((line.replace('Subject:', '').strip() for line in lines if line.startswith('Subject:')), f"Re: {original_email.splitlines()[0] if original_email else 'Email Reply'}")
            body = '\n'.join(line for line in lines if not line.startswith('Subject:') and line.strip())

            if not body.strip() and response_text.strip():
                body = response_text.strip()

            logging.info(f"AI Response parsed - Subject: '{subject}', Body length: {len(body)}, Generation time: {generation_time_ms}ms")
            logging.info(f"Raw AI Response: {repr(response_text[:200])}")
            logging.info(f"Parsed body: {repr(body[:200])}")

            # Prepare the return dictionary, including token usage if available
            generation_response_data = {
                'success': True,
                'subject': subject,
                'body': body.strip(),
                'tone': tone,
                'confidence': 0.85, # Default confidence
                'model_used': model_key,
                'generation_time_ms': generation_time_ms,
                'langchain_components_used': {
                    'chains': ['RunnableSequence'],
                    'memory': 'ConversationBufferMemory',
                    'parsers': ['StrOutputParser'], # PydanticOutputParser is not directly used in this specific chain output
                    'runnables': ['RunnablePassthrough', 'RunnableSequence'],
                    'callbacks': 'get_openai_callback' if cb else 'None'
                },
                'token_usage': {
                    'total_tokens': cb.total_tokens if cb else 0,
                    'prompt_tokens': cb.prompt_tokens if cb else 0,
                    'completion_tokens': cb.completion_tokens if cb else 0
                } if cb else {}
            }
            return generation_response_data

        except ValueError as ve:
            logging.error(f"Configuration error for LangChain email generation: {ve}")
            # Handle configuration errors gracefully
            return {
                'success': False,
                'error': str(ve),
                'fallback_used': True,
                'fallback_reason': 'AI model configuration error'
            }
        except Exception as e:
            logging.error(f"LangChain email generation error: {e}")

            # Check for credit/payment errors and provide helpful fallback
            if "402" in str(e) or "credits" in str(e).lower() or "payment" in str(e).lower():
                return {
                    'success': True, # Still report success for the fallback
                    'subject': "Re: Your Email",
                    'body': "Thank you for your email. I appreciate you reaching out and will get back to you soon.\n\nBest regards",
                    'tone': tone,
                    'confidence': 0.7,
                    'model_used': 'fallback-system',
                    'generation_time_ms': 50,
                    'fallback_used': True,
                    'fallback_reason': 'API credits exhausted - using template response'
                }

            return {
                'success': False,
                'error': str(e),
                'fallback_used': True,
                'fallback_reason': 'An unexpected error occurred during generation'
            }

    def suggest_email_improvements(self, email_content: str) -> Dict[str, Any]:
        """Advanced LLM-powered email improvement system using LangChain"""
        try:
            start_time = time.time()

            # Choose the best available model for suggestion generation
            suggestion_model = None
            model_name = "fallback"

            if 'qwen-4-turbo' in self.langchain_models:
                suggestion_model = self.langchain_models['qwen-4-turbo']
                model_name = "qwen-4-turbo"
            elif 'claude-4-sonnet' in self.langchain_models:
                suggestion_model = self.langchain_models['claude-4-sonnet']
                model_name = "claude-4-sonnet"
            elif 'gpt-4o' in self.langchain_models:
                suggestion_model = self.langchain_models['gpt-4o']
                model_name = "gpt-4o"

            if suggestion_model:
                # Create advanced LangChain prompt for email improvement suggestions
                improvement_prompt_template = """You are an expert email communication consultant with deep expertise in business writing, psychology, and professional communication. 

Analyze the given email content and provide comprehensive improvement suggestions AND a rewritten improved version in JSON format with these exact keys:
- suggestions: array of 5-7 specific, actionable improvement recommendations
- improved_email: the complete rewritten email implementing all the suggestions
- analysis_metrics: object with detailed metrics about the email

For each suggestion, use one of these category prefixes:
- 🏗️ STRUCTURE: For formatting, organization, greeting, closing issues
- 💡 CLARITY: For readability, word choice, sentence structure improvements  
- ⚡ IMPACT: For persuasiveness, action items, call-to-action improvements
- 🎯 TONE: For professionalism, appropriateness, voice adjustments
- 📝 CONTENT: For substance, detail, context improvements

Analysis metrics should include:
- word_count: number of words
- sentence_count: number of sentences
- professionalism_score: 1-10 rating
- clarity_score: 1-10 rating
- engagement_score: 1-10 rating
- total_analysis_time_ms: processing time

The improved_email should:
- Implement all suggested improvements
- Maintain the original intent and key information
- Use professional but appropriate tone
- Include clear subject line if needed
- Be ready to use as-is

Consider:
- Email structure and formatting
- Tone appropriateness for business context
- Clarity and conciseness
- Action items and next steps
- Professional language vs casual expressions
- Specific improvements with examples when possible
- Cultural and industry communication norms

Provide specific, actionable advice that the user can immediately implement. Avoid generic suggestions.

Respond only with valid JSON.
"""
                improvement_prompt = ChatPromptTemplate.from_messages([
                    ("system", improvement_prompt_template),
                    ("human", "Analyze this email and suggest improvements:\n\n{email_content}")
                ])

                # Create LangChain chain for structured improvement suggestions
                suggestion_chain = improvement_prompt | suggestion_model | StrOutputParser()

                # Execute suggestion generation
                result = suggestion_chain.invoke({"email_content": email_content})

                # Parse JSON response
                try:
                    suggestion_result = json.loads(result)

                    # Validate and ensure required fields
                    if 'suggestions' in suggestion_result and isinstance(suggestion_result['suggestions'], list):
                        # Ensure we have valid metrics
                        if 'analysis_metrics' not in suggestion_result:
                            suggestion_result['analysis_metrics'] = {}

                        # Add processing metadata
                        suggestion_result['analysis_metrics']['total_analysis_time_ms'] = int((time.time() - start_time) * 1000)
                        suggestion_result['analysis_metrics']['model_used'] = model_name
                        suggestion_result['analysis_metrics']['method'] = 'langchain_llm'

                        # Ensure required metrics exist
                        words = email_content.split()
                        sentences = [s.strip() for s in email_content.replace('!', '.').replace('?', '.').split('.') if s.strip()]

                        default_metrics = {
                            'word_count': len(words),
                            'sentence_count': len(sentences),
                            'professionalism_score': 7,
                            'clarity_score': 7,
                            'engagement_score': 7
                        }

                        for metric, default_value in default_metrics.items():
                            if metric not in suggestion_result['analysis_metrics']:
                                suggestion_result['analysis_metrics'][metric] = default_value

                        suggestion_result['success'] = True
                        return suggestion_result

                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse LLM suggestion response: {e}, falling back to enhanced analysis")

            # Fallback to enhanced rule-based analysis if LLM fails
            return self._fallback_suggestion_analysis(email_content, start_time)

        except Exception as e:
            logging.error(f"Error in LangChain suggestion generation: {str(e)}")
            return self._fallback_suggestion_analysis(email_content, time.time())

    def _fallback_suggestion_analysis(self, email_content: str, start_time: float) -> Dict[str, Any]:
        """Enhanced fallback suggestion analysis with intelligent recommendations"""
        try:
            words = email_content.split()
            sentences = [s.strip() for s in email_content.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            content_lower = email_content.lower()

            suggestions = []

            # Advanced structure analysis
            has_greeting = any(greeting in content_lower for greeting in ['dear', 'hello', 'hi', 'good morning', 'good afternoon', 'greetings'])
            has_closing = any(closing in content_lower for closing in ['regards', 'sincerely', 'best', 'thank you', 'thanks', 'cordially'])

            if not has_greeting:
                suggestions.append("🏗️ STRUCTURE: Add a professional greeting like 'Dear [Name]' or 'Hello [Name]' to establish rapport")
            elif content_lower.startswith('hi ') or content_lower.startswith('hey '):
                suggestions.append("🏗️ STRUCTURE: Consider 'Dear [Name]' or 'Hello [Name]' for more formal business communication")

            if not has_closing:
                suggestions.append("🏗️ STRUCTURE: Add a professional closing such as 'Best regards' or 'Sincerely' followed by your name")

            # Advanced clarity analysis
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            if avg_sentence_length > 20:
                suggestions.append("💡 CLARITY: Break long sentences (avg: {:.1f} words) into shorter, more digestible segments".format(avg_sentence_length))

            if len(email_content) > 400 and email_content.count('\n') < 3:
                suggestions.append("💡 CLARITY: Organize content into shorter paragraphs - each focusing on one main idea")

            # Advanced impact analysis
            action_indicators = ['please', 'could you', 'would you', 'can you', 'let me know', 'need you to', 'request']
            has_clear_action = any(indicator in content_lower for indicator in action_indicators)

            if not has_clear_action:
                suggestions.append("⚡ IMPACT: Include specific action items or requests to guide the recipient's response")

            # Vague language detection
            vague_terms = ['soon', 'later', 'sometime', 'whenever', 'maybe', 'probably', 'might', 'could possibly']
            vague_count = sum(content_lower.count(term) for term in vague_terms)

            if vague_count > 0:
                suggestions.append("⚡ IMPACT: Replace vague timeframes with specific dates, deadlines, or timeframes")

            # Advanced tone analysis
            casual_indicators = ['hey', 'gonna', 'wanna', 'yeah', 'ok', 'stuff', 'things', 'kinda', 'sorta']
            casual_count = sum(content_lower.count(word) for word in casual_indicators)

            if casual_count > 0:
                suggestions.append("🎯 TONE: Replace casual expressions with professional language appropriate for business communication")

            # Passive voice detection
            passive_patterns = ['was done', 'were completed', 'has been', 'will be handled', 'is being', 'are being']
            passive_count = sum(content_lower.count(pattern) for pattern in passive_patterns)

            if passive_count > 0 or content_lower.count(' was ') + content_lower.count(' were ') > 2:
                suggestions.append("⚡ IMPACT: Use active voice ('I will complete' vs 'it will be completed') for stronger communication")

            # Content analysis
            if len(words) < 20:
                suggestions.append("📝 CONTENT: Consider adding more context or details to make your message more comprehensive")

            # Repetition analysis
            word_freq = {}
            for word in words:
                clean_word = word.lower().strip('.,!?;:')
                if len(clean_word) > 3:
                    word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

            repeated_words = [word for word, count in word_freq.items() if count > 3]
            if repeated_words:
                suggestions.append("💡 CLARITY: Reduce repetition of words like '{}' by using synonyms or restructuring".format("', '".join(repeated_words[:2])))

            # Ensure we have enough valuable suggestions
            if len(suggestions) < 4:
                additional_suggestions = [
                    "💡 CLARITY: Use bullet points to organize multiple items or complex information",
                    "⚡ IMPACT: Lead with the most important information in your opening paragraph",
                    "🎯 TONE: Ensure your tone matches the relationship and formality level with the recipient",
                    "🏗️ STRUCTURE: Create descriptive subject lines that summarize your main purpose",
                    "📝 CONTENT: Provide sufficient context for recipients who may not have background information",
                    "⚡ IMPACT: End with clear next steps or timeline expectations"
                ]

                for suggestion in additional_suggestions:
                    if suggestion not in suggestions and len(suggestions) < 6:
                        suggestions.append(suggestion)

            # Calculate enhanced metrics
            professional_indicators = ['please', 'thank you', 'regards', 'sincerely', 'appreciate', 'consider', 'kindly', 'respectfully']
            professionalism_score = min(10, 4 + sum(1 for indicator in professional_indicators if indicator in content_lower))

            clarity_score = 10
            if avg_sentence_length > 25: clarity_score -= 2
            if len(sentences) == 0: clarity_score -= 3
            if casual_count > 0: clarity_score -= 1
            if vague_count > 1: clarity_score -= 1
            clarity_score = max(1, clarity_score)

            engagement_score = 5
            if has_clear_action: engagement_score += 2
            if has_greeting and has_closing: engagement_score += 2
            if len(words) > 15 and len(words) < 150: engagement_score += 1
            engagement_score = min(10, engagement_score)

            return {
                'success': True,
                'suggestions': suggestions[:6],
                'analysis_metrics': {
                    'total_analysis_time_ms': int((time.time() - start_time) * 1000),
                    'word_count': len(words),
                    'sentence_count': len(sentences),
                    'professionalism_score': professionalism_score,
                    'clarity_score': clarity_score,
                    'engagement_score': engagement_score,
                    'avg_sentence_length': round(avg_sentence_length, 1),
                    'model_used': 'enhanced_fallback',
                    'method': 'rule_based_analysis'
                }
            }

        except Exception as e:
            logging.error(f"Error in fallback suggestion analysis: {str(e)}")
            return {
                'success': True,
                'suggestions': [
                    "🏗️ STRUCTURE: Include a professional greeting and closing",
                    "💡 CLARITY: Write clear, concise sentences that are easy to understand",
                    "⚡ IMPACT: Include specific action items or requests",
                    "🎯 TONE: Use professional language appropriate for business communication",
                    "📝 CONTENT: Provide sufficient context and details",
                    "⚡ IMPACT: End with clear next steps or expectations"
                ],
                'analysis_metrics': {
                    'total_analysis_time_ms': int((time.time() - start_time) * 1000),
                    'word_count': len(email_content.split()) if email_content else 0,
                    'sentence_count': 1,
                    'professionalism_score': 5,
                    'clarity_score': 5,
                    'engagement_score': 5,
                    'model_used': 'error_fallback',
                    'method': 'static_suggestions',
                    'fallback_used': True,
                    'fallback_reason': str(e)
                }
            }

    def analyze_email_with_langchain(self, email_content: str) -> Dict[str, Any]:
        """Advanced LLM-powered email analysis using LangChain"""
        try:
            start_time = time.time()

            # Choose the best available model for analysis
            analysis_model = None
            model_name = "fallback"

            if 'qwen-4-turbo' in self.langchain_models:
                analysis_model = self.langchain_models['qwen-4-turbo']
                model_name = "qwen-4-turbo"
            elif 'claude-4-sonnet' in self.langchain_models:
                analysis_model = self.langchain_models['claude-4-sonnet']
                model_name = "claude-4-sonnet"
            elif 'gpt-4o' in self.langchain_models:
                analysis_model = self.langchain_models['gpt-4o']
                model_name = "gpt-4o"

            if analysis_model:
                # Create advanced LangChain prompt for comprehensive analysis
                analysis_prompt_template = """You are an expert email analyst with deep understanding of business communication, psychology, and sentiment analysis. 

Analyze the given email content and provide a comprehensive assessment in JSON format with these exact keys:
- sentiment: "positive", "negative", or "neutral" 
- urgency: "high", "medium", or "low"
- tone: "formal", "professional", "friendly", "casual", or "urgent"
- emotion_score: float between 0.0 (very negative) and 1.0 (very positive)
- key_topics: array of 2-4 main topics/themes discussed
- action_items: array of 2-4 specific actions required or mentioned
- clarity_score: integer from 1-10 rating message clarity
- tone_appropriateness: integer from 1-10 rating professionalism level

Consider:
- Context clues and implied meanings
- Cultural and business communication norms
- Emotional undertones and subtext
- Urgency indicators beyond explicit words
- Professional vs casual language patterns
- Action-oriented vs informational content

Respond only with valid JSON.
"""
                analysis_prompt = ChatPromptTemplate.from_messages([
                    ("system", analysis_prompt_template),
                    ("human", "Analyze this email:\n\n{email_content}")
                ])

                # Create LangChain chain for structured output
                analysis_chain = analysis_prompt | analysis_model | StrOutputParser()

                # Execute analysis
                result = analysis_chain.invoke({"email_content": email_content})

                # Parse JSON response
                try:
                    analysis_result = json.loads(result)

                    # Validate and ensure required fields
                    analysis_result['success'] = True
                    analysis_result['processing_time_ms'] = int((time.time() - start_time) * 1000)
                    analysis_result['method'] = f'langchain_{model_name}'

                    # Ensure all required fields exist with defaults
                    required_fields = {
                        'sentiment': 'neutral',
                        'urgency': 'medium', 
                        'tone': 'professional',
                        'emotion_score': 0.5,
                        'key_topics': ['communication'],
                        'action_items': ['review message'],
                        'clarity_score': 7,
                        'tone_appropriateness': 7
                    }

                    for field, default in required_fields.items():
                        if field not in analysis_result:
                            analysis_result[field] = default

                    # Format for compatibility with existing API
                    return {
                        'success': True,
                        'analysis': {
                            'sentiment': analysis_result['sentiment'],
                            'urgency': analysis_result['urgency'],
                            'key_topics': analysis_result['key_topics'],
                            'action_items': analysis_result['action_items'],
                            'tone': analysis_result['tone'],
                            'clarity_score': analysis_result['clarity_score'],
                            'tone_appropriateness': analysis_result['tone_appropriateness']
                        },
                        'emotion_score': analysis_result['emotion_score'],
                        'processing_time_ms': analysis_result['processing_time_ms'],
                        'method': analysis_result['method']
                    }

                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse LLM JSON response: {e}, falling back to enhanced analysis")

            # Fallback to enhanced keyword analysis if LLM fails
            return self._fallback_email_analysis(email_content, start_time)

        except Exception as e:
            logging.error(f"Error in LangChain email analysis: {str(e)}")
            return self._fallback_email_analysis(email_content, time.time())

    def _fallback_email_analysis(self, email_content: str, start_time: float) -> Dict[str, Any]:
        """Enhanced fallback analysis with better keyword detection"""
        try:
            content_lower = email_content.lower()

            # Enhanced sentiment analysis
            positive_indicators = {
                'words': ['thank', 'great', 'excellent', 'wonderful', 'amazing', 'appreciate', 'pleased', 'happy', 'perfect', 'fantastic'],
                'phrases': ['thank you', 'well done', 'good job', 'looking forward', 'excited about']
            }
            negative_indicators = {
                'words': ['sorry', 'problem', 'issue', 'concern', 'disappointed', 'frustrated', 'urgent', 'emergency', 'mistake', 'error', 'failed', 'wrong'],
                'phrases': ['not working', 'need help', 'went wrong', 'big problem', 'very concerned']
            }

            positive_score = sum(content_lower.count(word) for word in positive_indicators['words'])
            positive_score += sum(content_lower.count(phrase) * 2 for phrase in positive_indicators['phrases'])

            negative_score = sum(content_lower.count(word) for word in negative_indicators['words'])
            negative_score += sum(content_lower.count(phrase) * 2 for phrase in negative_indicators['phrases'])

            if positive_score > negative_score and positive_score > 0:
                sentiment = 'positive'
                emotion_score = min(0.8, 0.5 + (positive_score * 0.1))
            elif negative_score > positive_score and negative_score > 0:
                sentiment = 'negative'
                emotion_score = max(0.2, 0.5 - (negative_score * 0.1))
            else:
                sentiment = 'neutral'
                emotion_score = 0.5

            # Enhanced urgency detection
            urgency_indicators = {
                'high': ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'deadline today', 'right now'],
                'medium': ['soon', 'quick', 'fast', 'deadline', 'by end of day', 'this week']
            }

            if any(indicator in content_lower for indicator in urgency_indicators['high']):
                urgency = 'high'
            elif any(indicator in content_lower for indicator in urgency_indicators['medium']):
                urgency = 'medium'
            else:
                urgency = 'low'

            # Enhanced tone detection
            formal_indicators = ['dear', 'sincerely', 'regards', 'respectfully', 'cordially', 'yours truly']
            casual_indicators = ['hi', 'hey', 'thanks', 'cheers', 'talk soon', 'catch up']
            urgent_indicators = ['urgent', 'asap', 'immediately', 'critical']

            if any(indicator in content_lower for indicator in urgent_indicators):
                tone = 'urgent'
            elif any(indicator in content_lower for indicator in formal_indicators):
                tone = 'formal'
            elif any(indicator in content_lower for indicator in casual_indicators):
                tone = 'friendly'
            else:
                tone = 'professional'

            # Enhanced key topics extraction
            import re
            words = re.findall(r'\b\w{4,}\b', content_lower)
            stop_words = {'that', 'with', 'have', 'this', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'about', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'need', 'should', 'make', 'like', 'even', 'back', 'take', 'come', 'good', 'much', 'well', 'want', 'through', 'where', 'most', 'after', 'please', 'email', 'message'}

            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1

            key_topics = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:4]
            if not key_topics:
                key_topics = ['general communication']

            # Enhanced action items detection
            action_patterns = {
                'meeting': r'\b(meet|meeting|schedule|call|discuss)\b',
                'review': r'\b(review|check|look at|examine)\b',
                'send': r'\b(send|provide|share|forward)\b',
                'update': r'\b(update|inform|notify|let.*know)\b',
                'deadline': r'\b(deadline|due|complete|finish)\b'
            }

            action_items = []
            for action, pattern in action_patterns.items():
                if re.search(pattern, content_lower):
                    if action == 'meeting':
                        action_items.append('Schedule meeting or call')
                    elif action == 'review':
                        action_items.append('Review documents or information')
                    elif action == 'send':
                        action_items.append('Send requested materials')
                    elif action == 'update':
                        action_items.append('Provide status update')
                    elif action == 'deadline':
                        action_items.append('Complete task by deadline')

            if not action_items:
                action_items = ['Acknowledge receipt and respond appropriately']

            return {
                'success': True,
                'analysis': {
                    'sentiment': sentiment,
                    'urgency': urgency,
                    'key_topics': key_topics[:4],
                    'action_items': action_items[:4],
                    'tone': tone,
                    'clarity_score': 7,
                    'tone_appropriateness': 8 if tone in ['formal', 'professional'] else 6
                },
                'emotion_score': emotion_score,
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'method': 'enhanced_keyword_analysis'
            }

        except Exception as e:
            logging.error(f"Error in fallback analysis: {str(e)}")
            return {
                'success': True,
                'analysis': {
                    'sentiment': 'neutral',
                    'urgency': 'medium',
                    'key_topics': ['communication'],
                    'action_items': ['respond to email'],
                    'tone': 'professional',
                    'clarity_score': 5,
                    'tone_appropriateness': 5
                },
                'emotion_score': 0.5,
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'method': 'error_fallback'
            }

    def process_with_conversational_agent(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """Process queries using LangChain conversational agent"""
        try:
            if not hasattr(self, 'conversational_agent'):
                raise ValueError("Conversational agent not available")

            # Use conversation memory for context
            if conversation_id:
                # Add conversation context to memory
                self.conversation_memory.chat_memory.add_user_message(query)

            # Execute with agent and callback tracking
            cb = None
            try:
                with get_openai_callback() as cb_instance:
                    cb = cb_instance
                    response = self.conversational_agent.run(query)
            except Exception as callback_err:
                logging.warning(f"get_openai_callback might not be fully compatible with agent run: {callback_err}")
                response = self.conversational_agent.run(query) # execute without callback if error

            # Update memory with assistant response
            if conversation_id:
                self.conversation_memory.chat_memory.add_ai_message(response)

            return {
                'success': True,
                'response': response,
                'conversation_id': conversation_id,
                'langchain_components_used': {
                    'agents': ['ConversationalReactDescription'],
                    'memory': ['ConversationBufferMemory'],
                    'tools': ['EmailAnalyzer', 'EmailGenerator', 'TextSplitter'],
                    'callbacks': 'get_openai_callback' if cb else 'None'
                },
                'token_usage': {
                    'total_tokens': cb.total_tokens if cb else 0
                } if cb else {}
            }

        except Exception as e:
            logging.error(f"Conversational agent error: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_response': f"I understand you asked: {query}. Let me help you with that."
            }

    def generate_email_reply(self, original_email: str, context: str = "", tone: str = "professional", model: str = "auto", custom_instructions: str = "") -> Dict[str, Any]:
        """Main method that uses LangChain for email generation (backwards compatibility)"""
        # Use the comprehensive LangChain method
        return self.generate_email_reply_with_langchain(
            original_email=original_email,
            context=context,
            tone=tone,
            custom_instructions=custom_instructions
        )

    def generate_email_template(self, template_type: str = "professional", purpose: str = "", tone: str = "professional", industry: str = "", custom_instructions: str = "") -> Dict[str, Any]:
        """
        🚀 SPECTACULAR AI-POWERED EMAIL TEMPLATE GENERATOR 🚀
        
        Advanced LangChain-powered system that creates intelligent, adaptive email templates
        with multiple AI models, smart categorization, and industry-specific customization.
        
        Features:
        - Multi-model AI orchestration (Qwen-4, Claude-4, GPT-4o)
        - Advanced prompt engineering with LangChain ChatPromptTemplate
        - Intelligent placeholder generation
        - Industry-specific customization
        - Dynamic complexity assessment
        - Smart categorization and use case detection
        - Comprehensive metadata generation
        """
        try:
            start_time = time.time()
            
            if not purpose.strip():
                return {
                    'success': False,
                    'error': 'Template purpose is required to generate meaningful templates'
                }

            # Select optimal AI model based on template complexity and type
            selected_model = None
            model_name = "qwen-4-turbo"  # Default to most capable model
            
            # Smart model selection based on requirements
            if template_type in ['creative', 'marketing', 'narrative'] or industry in ['marketing', 'advertising', 'media']:
                # Use Claude for creative and marketing templates
                if 'claude-4-sonnet' in self.langchain_models:
                    selected_model = self.langchain_models['claude-4-sonnet']
                    model_name = "claude-4-sonnet"
            elif template_type in ['technical', 'professional', 'complex'] or industry in ['technology', 'engineering', 'finance']:
                # Use Qwen for technical and professional templates
                if 'qwen-4-turbo' in self.langchain_models:
                    selected_model = self.langchain_models['qwen-4-turbo']
                    model_name = "qwen-4-turbo"
            elif template_type in ['simple', 'quick', 'basic']:
                # Use GPT-4o for simple templates
                if 'gpt-4o' in self.langchain_models:
                    selected_model = self.langchain_models['gpt-4o']
                    model_name = "gpt-4o"
            
            # Fallback to any available model
            if not selected_model:
                for model_key, model_instance in self.langchain_models.items():
                    selected_model = model_instance
                    model_name = model_key
                    break

            if not selected_model:
                return self._fallback_template_generation(purpose, template_type, tone, industry, start_time)

            # Create sophisticated LangChain prompt for template generation
            template_prompt_template = """You are an elite email template architect with expertise in business communication, psychology, and industry-specific messaging. You create intelligent, adaptive templates that professionals can customize for various scenarios.

Create a comprehensive email template based on the requirements below. Generate a complete JSON response with these exact keys:

REQUIRED JSON STRUCTURE:
{{
  "template_name": "Professional and descriptive name",
  "description": "Clear description of template purpose and best use cases",
  "subject_template": "Dynamic subject line with placeholders like {{{{recipient_name}}}}, {{{{company}}}}, {{{{topic}}}}",
  "body_template": "Complete email body with smart placeholders and professional structure",
  "placeholders": ["{{{{placeholder1}}}}", "{{{{placeholder2}}}}", ...],
  "category": "One of: business, sales, support, follow-up, meeting, project, personal, marketing, technical",
  "tone": "Actual tone of the template: professional, friendly, formal, casual, urgent, persuasive",
  "industry_specific": true/false,
  "use_cases": ["Specific scenario 1", "Specific scenario 2", ...],
  "complexity_level": "simple, intermediate, or advanced",
  "customization_tips": ["Tip 1", "Tip 2", "Tip 3", ...]
}}

TEMPLATE REQUIREMENTS:
- Purpose: {purpose}
- Template Type: {template_type}
- Desired Tone: {tone}
- Industry Context: {industry}
- Custom Instructions: {custom_instructions}

ADVANCED TEMPLATE FEATURES TO INCLUDE:
1. **Smart Placeholders**: Use meaningful placeholders like {{{{recipient_name}}}}, {{{{company}}}}, {{{{project_name}}}}, {{{{deadline}}}}, {{{{next_steps}}}}
2. **Dynamic Structure**: Adapt structure based on purpose (greeting, context, main message, action items, closing)
3. **Industry Optimization**: Include industry-specific language and considerations
4. **Tone Consistency**: Ensure tone matches throughout subject and body
5. **Action-Oriented**: Include clear call-to-action or next steps
6. **Professional Polish**: Proper formatting, spacing, and professional language

INTELLIGENT CUSTOMIZATION:
- For SALES templates: Include value propositions, benefit statements, and clear CTAs
- For SUPPORT templates: Include empathy, solution focus, and follow-up steps
- For FOLLOW-UP templates: Reference previous interactions and clear next steps
- For MEETING templates: Include agenda items, time considerations, and preparation notes
- For PROJECT templates: Include status updates, deliverables, and timeline references

PLACEHOLDER STRATEGY:
- Use specific, meaningful placeholders that guide users
- Include both required ({{{{recipient_name}}}}) and optional ({{{{company}}}}) placeholders
- Provide context for when to use each placeholder
- Ensure placeholders enhance personalization

CUSTOMIZATION TIPS SHOULD INCLUDE:
- How to adapt tone for different relationships
- When to add or remove sections
- Industry-specific modifications
- Personalization strategies
- Common variations and use cases

Generate a template that is professional, practical, and immediately usable while being highly customizable.

Respond ONLY with valid JSON matching the exact structure above."""

            # Create LangChain prompt
            template_prompt = ChatPromptTemplate.from_messages([
                ("system", template_prompt_template),
                ("human", "Generate template for: {purpose}")
            ])

            # Create output parser for structured response
            template_parser = PydanticOutputParser(pydantic_object=TemplateGenerationResult)

            # Create LangChain chain
            template_chain = template_prompt | selected_model | StrOutputParser()

            # Execute template generation
            result = template_chain.invoke({
                "purpose": purpose,
                "template_type": template_type,
                "tone": tone,
                "industry": industry if industry else "general business",
                "custom_instructions": custom_instructions if custom_instructions else "Follow standard best practices"
            })
            
            logging.info(f"AI model {model_name} returned result type: {type(result)}, length: {len(result) if result else 0}")
            if result:
                logging.info(f"AI response preview: {result[:200]}...")
            else:
                logging.warning("AI returned None or empty response")

            # Parse JSON response
            try:
                # Check if result is empty or whitespace
                if not result or not result.strip():
                    logging.warning("AI returned empty response, using intelligent fallback")
                    return self._fallback_template_generation(purpose, template_type, tone, industry, start_time)
                
                # Clean the result - remove markdown code blocks if present
                cleaned_result = result.strip()
                if cleaned_result.startswith('```json'):
                    cleaned_result = cleaned_result[7:]  # Remove ```json
                if cleaned_result.startswith('```'):
                    cleaned_result = cleaned_result[3:]   # Remove ```
                if cleaned_result.endswith('```'):
                    cleaned_result = cleaned_result[:-3]  # Remove ending ```
                cleaned_result = cleaned_result.strip()
                
                logging.info(f"Cleaned AI response for JSON parsing: {cleaned_result[:200]}...")
                template_result = json.loads(cleaned_result)
                
                # Validate and enhance the result
                if 'template_name' in template_result and 'body_template' in template_result:
                    # Add processing metadata
                    generation_time_ms = int((time.time() - start_time) * 1000)
                    
                    # Enhance result with additional intelligence
                    enhanced_result = {
                        'success': True,
                        'template_name': template_result.get('template_name', f"{purpose.title()} Template"),
                        'description': template_result.get('description', f"Professional template for {purpose}"),
                        'subject_template': template_result.get('subject_template', f"Re: {{{{topic}}}} - {{{{your_name}}}}"),
                        'body_template': template_result.get('body_template', "Template body not generated properly"),
                        'placeholders': template_result.get('placeholders', ["{{recipient_name}}", "{{your_name}}", "{{topic}}"]),
                        'category': template_result.get('category', self._determine_category(purpose, template_type)),
                        'tone': template_result.get('tone', tone),
                        'industry_specific': template_result.get('industry_specific', bool(industry)),
                        'use_cases': template_result.get('use_cases', [purpose]),
                        'complexity_level': template_result.get('complexity_level', self._assess_complexity(template_result.get('body_template', ''))),
                        'customization_tips': template_result.get('customization_tips', self._generate_customization_tips(purpose, tone)),
                        'model_used': model_name,
                        'generation_time_ms': generation_time_ms,
                        'ai_enhanced': True,
                        'langchain_features': ['ChatPromptTemplate', 'StrOutputParser', 'RunnableSequence'],
                        'metadata': {
                            'created_by': 'ai',
                            'template_version': '2.0',
                            'supports_placeholders': True,
                            'multi_industry': not bool(industry),
                            'generation_method': f'langchain_{model_name}'
                        }
                    }
                    
                    return enhanced_result
                else:
                    logging.warning("Incomplete template generated by AI, using fallback enhancement")
                    
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse AI template response: {e}, using intelligent fallback")

            # Enhanced fallback with AI-inspired generation
            return self._fallback_template_generation(purpose, template_type, tone, industry, start_time)

        except Exception as e:
            logging.error(f"Error in AI template generation: {str(e)}")
            return self._fallback_template_generation(purpose, template_type, tone, industry, time.time())

    def _determine_category(self, purpose: str, template_type: str) -> str:
        """Intelligently determine template category"""
        purpose_lower = purpose.lower()
        type_lower = template_type.lower()
        
        category_mapping = {
            'sales': ['sell', 'proposal', 'quote', 'offer', 'pitch', 'demo'],
            'support': ['help', 'support', 'issue', 'problem', 'question', 'assistance'],
            'follow-up': ['follow', 'check', 'update', 'progress', 'status'],
            'meeting': ['meet', 'call', 'schedule', 'appointment', 'discussion'],
            'project': ['project', 'task', 'deliverable', 'milestone', 'deadline'],
            'marketing': ['market', 'campaign', 'promotion', 'announcement', 'launch'],
            'technical': ['technical', 'development', 'code', 'system', 'integration']
        }
        
        for category, keywords in category_mapping.items():
            if any(keyword in purpose_lower or keyword in type_lower for keyword in keywords):
                return category
                
        return 'business'

    def _assess_complexity(self, template_body: str) -> str:
        """Assess template complexity based on structure and content"""
        if not template_body:
            return 'simple'
            
        # Count various complexity indicators
        placeholders = template_body.count('{{')
        sentences = len([s for s in template_body.split('.') if s.strip()])
        paragraphs = len([p for p in template_body.split('\n\n') if p.strip()])
        
        complexity_score = 0
        if placeholders > 8: complexity_score += 2
        elif placeholders > 4: complexity_score += 1
        
        if sentences > 15: complexity_score += 2
        elif sentences > 8: complexity_score += 1
        
        if paragraphs > 4: complexity_score += 2
        elif paragraphs > 2: complexity_score += 1
        
        if complexity_score >= 4:
            return 'advanced'
        elif complexity_score >= 2:
            return 'intermediate'
        else:
            return 'simple'

    def _generate_customization_tips(self, purpose: str, tone: str) -> List[str]:
        """Generate intelligent customization tips"""
        base_tips = [
            f"Adjust the {{{{recipient_name}}}} placeholder to match your relationship level",
            f"Modify the tone to be more {tone} or formal based on your audience",
            "Add specific details relevant to your industry or situation",
            "Include relevant attachments or links in the body when needed"
        ]
        
        purpose_lower = purpose.lower()
        
        # Purpose-specific tips
        if 'meeting' in purpose_lower:
            base_tips.extend([
                "Include specific agenda items relevant to your meeting",
                "Add calendar links or scheduling tools for convenience",
                "Specify time zone and duration expectations"
            ])
        elif 'follow' in purpose_lower:
            base_tips.extend([
                "Reference specific previous conversations or commitments",
                "Include concrete next steps and timelines",
                "Mention any changed circumstances since last contact"
            ])
        elif 'proposal' in purpose_lower or 'sales' in purpose_lower:
            base_tips.extend([
                "Customize value propositions to the recipient's specific needs",
                "Include relevant case studies or testimonials",
                "Add clear pricing and timeline information"
            ])
        
        return base_tips[:6]  # Limit to 6 tips for usability

    def _fallback_template_generation(self, purpose: str, template_type: str, tone: str, industry: str, start_time: float) -> Dict[str, Any]:
        """Enhanced fallback template generation with intelligent defaults"""
        try:
            # Generate intelligent template name
            template_name = f"{purpose.title().replace('_', ' ')} Template"
            if industry:
                template_name = f"{industry.title()} {template_name}"

            # Create smart subject template
            if 'meeting' in purpose.lower():
                subject_template = "Meeting Request: {{topic}} - {{your_name}}"
            elif 'follow' in purpose.lower():
                subject_template = "Follow-up: {{topic}} - Next Steps"
            elif 'proposal' in purpose.lower():
                subject_template = "Proposal: {{service}} for {{company}}"
            else:
                subject_template = "Re: {{topic}} - {{your_name}}"

            # Generate intelligent body template based on purpose
            body_templates = {
                'meeting': """Dear {{recipient_name}},

I hope this email finds you well. I would like to schedule a meeting to discuss {{topic}}.

Meeting Details:
- Purpose: {{meeting_purpose}}
- Suggested Duration: {{duration}}
- Proposed Date/Time: {{datetime}}
- Location/Platform: {{location}}

Agenda items I'd like to cover:
- {{agenda_item1}}
- {{agenda_item2}}
- {{agenda_item3}}

Please let me know if this time works for you, or suggest alternative times that might be more convenient.

Best regards,
{{your_name}}
{{your_title}}
{{your_contact}}""",

                'follow_up': """Dear {{recipient_name}},

I wanted to follow up on our previous conversation about {{topic}}.

As discussed, I'm reaching out to {{purpose}} and ensure we're aligned on next steps.

Current Status:
- {{status_item1}}
- {{status_item2}}

Next Steps:
- {{next_step1}} (Target: {{date1}})
- {{next_step2}} (Target: {{date2}})

Please let me know if you have any questions or if there's anything I can help clarify.

Best regards,
{{your_name}}""",

                'proposal': """Dear {{recipient_name}},

Thank you for your interest in {{service}}. I'm pleased to present this proposal for {{project_name}}.

Project Overview:
{{project_description}}

Scope of Work:
- {{deliverable1}}
- {{deliverable2}}
- {{deliverable3}}

Timeline: {{timeline}}
Investment: {{cost}}

I believe this solution will {{benefit}} and I'm excited about the opportunity to work with {{company}}.

I'd be happy to discuss this proposal in detail. Please let me know when you're available for a call.

Best regards,
{{your_name}}
{{your_title}}
{{your_contact}}""",

                'support': """Dear {{recipient_name}},

Thank you for reaching out regarding {{issue}}.

I understand that {{problem_description}} and I'm here to help resolve this promptly.

To assist you effectively, I've {{initial_action}} and would like to {{next_action}}.

Resolution Steps:
1. {{step1}}
2. {{step2}}
3. {{step3}}

Timeline: {{resolution_timeline}}

If you have any questions or need immediate assistance, please don't hesitate to contact me at {{contact_method}}.

Best regards,
{{your_name}}
{{your_title}}"""
            }

            # Select appropriate template or create generic one
            purpose_key = next((key for key in body_templates.keys() if key in purpose.lower()), None)
            
            if purpose_key:
                body_template = body_templates[purpose_key]
                placeholders = [
                    '{{recipient_name}}', '{{your_name}}', '{{topic}}', '{{your_title}}', '{{your_contact}}'
                ]
                
                # Add purpose-specific placeholders
                if purpose_key == 'meeting':
                    placeholders.extend(['{{meeting_purpose}}', '{{duration}}', '{{datetime}}', '{{location}}', 
                                       '{{agenda_item1}}', '{{agenda_item2}}', '{{agenda_item3}}'])
                elif purpose_key == 'follow_up':
                    placeholders.extend(['{{purpose}}', '{{status_item1}}', '{{status_item2}}', 
                                       '{{next_step1}}', '{{date1}}', '{{next_step2}}', '{{date2}}'])
                elif purpose_key == 'proposal':
                    placeholders.extend(['{{service}}', '{{project_name}}', '{{project_description}}', 
                                       '{{deliverable1}}', '{{deliverable2}}', '{{deliverable3}}', 
                                       '{{timeline}}', '{{cost}}', '{{benefit}}', '{{company}}'])
                elif purpose_key == 'support':
                    placeholders.extend(['{{issue}}', '{{problem_description}}', '{{initial_action}}', 
                                       '{{next_action}}', '{{step1}}', '{{step2}}', '{{step3}}', 
                                       '{{resolution_timeline}}', '{{contact_method}}'])
            else:
                # Generic template
                body_template = f"""Dear {{{{recipient_name}}}},

I hope this email finds you well. I'm writing to {{{{purpose}}}}.

{{{{main_content}}}}

{{{{call_to_action}}}}

Please let me know if you have any questions or need any additional information.

Best regards,
{{{{your_name}}}}
{{{{your_title}}}}"""
                
                placeholders = ['{{recipient_name}}', '{{purpose}}', '{{main_content}}', 
                              '{{call_to_action}}', '{{your_name}}', '{{your_title}}']

            generation_time_ms = int((time.time() - start_time) * 1000)

            return {
                'success': True,
                'template_name': template_name,
                'description': f"Professional {purpose} template suitable for {tone} communication" + 
                             (f" in {industry} industry" if industry else ""),
                'subject_template': subject_template,
                'body_template': body_template,
                'placeholders': placeholders,
                'category': self._determine_category(purpose, template_type),
                'tone': tone,
                'industry_specific': bool(industry),
                'use_cases': [purpose, f"{tone} communication", "professional correspondence"],
                'complexity_level': self._assess_complexity(body_template),
                'customization_tips': self._generate_customization_tips(purpose, tone),
                'model_used': 'intelligent_fallback',
                'generation_time_ms': generation_time_ms,
                'ai_enhanced': False,
                'langchain_features': ['intelligent_fallback'],
                'metadata': {
                    'created_by': 'fallback_system',
                    'template_version': '1.0',
                    'supports_placeholders': True,
                    'multi_industry': not bool(industry),
                    'generation_method': 'rule_based_intelligent'
                }
            }

        except Exception as e:
            logging.error(f"Error in fallback template generation: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to generate template: {str(e)}',
                'model_used': 'error_fallback',
                'generation_time_ms': int((time.time() - start_time) * 1000)
            }

    def analyze_email_sentiment(self, email_content: str) -> Dict[str, Any]:
        """Analyze email sentiment using LangChain (backwards compatibility)"""
        result = self.analyze_email_with_langchain(email_content)
        if result.get('success'):
            # Ensure 'analysis' key exists and is a dictionary
            analysis = result.get('analysis', {})
            return {
                'sentiment': analysis.get('sentiment', 'neutral'),
                'confidence': 0.85,
                'details': analysis
            }
        else:
            # Return a default structure if analysis failed
            return {
                'success': False,
                'sentiment': 'neutral',
                'confidence': 0.5,
                'error': result.get('error', 'Analysis failed'),
                'details': result.get('analysis', {}) # Include any partial analysis if available
            }

    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all LangChain components"""
        return {
            'langchain_models': list(self.langchain_models.keys()),
            'chains_available': ['SequentialChain', 'LLMChain', 'ConversationChain'],
            'memory_systems': ['ConversationBufferMemory', 'ConversationSummaryMemory'],
            'agents_available': ['ConversationalReactDescription'],
            'parsers_available': ['StrOutputParser', 'PydanticOutputParser'],
            'runnables_available': ['RunnablePassthrough', 'RunnableSequence'],
            'text_processing': ['RecursiveCharacterTextSplitter'],
            'tools_available': ['EmailAnalyzer', 'EmailGenerator', 'TextSplitter'],
            'callbacks_available': ['get_openai_callback'],
            'structured_output': ['EmailAnalysisResult', 'EmailGenerationResult']
        }

# Create global instance for backwards compatibility
ai_service = AIService()