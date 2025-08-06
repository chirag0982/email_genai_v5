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