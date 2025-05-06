import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from utils.db_utils import add_message_to_db, get_conversation_messages
from utils.gemini_utils import get_ai_response as gemini_get_response, analyze_image_content as gemini_analyze_image, get_embedding as gemini_get_embedding
from utils.openai_utils import get_ai_response as openai_get_response, analyze_image_content as openai_analyze_image, get_embedding as openai_get_embedding
from utils.gemini_langchain_utils import get_rag_response as gemini_get_rag_response, get_multimodal_response as gemini_get_multimodal_response
from utils.langchain_utils import get_rag_response as openai_get_rag_response, get_multimodal_response as openai_get_multimodal_response
from utils.vector_store import search_vector_store
from utils.document_utils import get_document_summary
from utils.agent_reasoning import AgentMemory, ReasoningEngine, AgentTools

# Set up logging
logger = logging.getLogger(__name__)

class Agent:
    """
    Enhanced Agent class with agentic capabilities for conversation interactions
    with various LLMs and multimodal inputs.
    """
    
    def __init__(self):
        """Initialize the Agent with default values and reasoning capabilities."""
        # Core attributes
        self.conversation_id = None
        self.last_message_id = None
        self.selected_model = "gemini"  # default model
        self.model_version = None  # specific version of the model (e.g., "gpt-4o" or "gemini-1.0-pro")
        self.api_key = None
        
        # Agent capabilities 
        self.memory = AgentMemory()
        self.reasoning = ReasoningEngine()
        self.tools = AgentTools()
        
        # Agent state
        self.last_query_type = None  # track the type of the last query
        self.last_plan = []          # last execution plan
        self.last_fallback = None    # fallback used (if any)
        self.performance_metrics = {
            "successful_responses": 0,
            "failed_responses": 0,
            "rag_hits": 0,
            "image_analyses": 0,
            "document_analyses": 0
        }
    
    def set_conversation_id(self, conversation_id):
        """Set the conversation ID for this agent instance."""
        self.conversation_id = conversation_id
        
    def set_model(self, model_name, api_key=None, model_version=None):
        """
        Set which AI model to use.
        
        Args:
            model_name (str): 'gemini' or 'openai'
            api_key (str, optional): API key for the selected model
            model_version (str, optional): Specific version of the model to use
        """
        self.selected_model = model_name
        self.api_key = api_key
        self.model_version = model_version
    
    def get_conversation_history(self, limit=20):
        """
        Get the conversation history.
        
        Args:
            limit (int): Maximum number of messages to retrieve
            
        Returns:
            list: List of message dictionaries
        """
        if not self.conversation_id:
            return []
        
        return get_conversation_messages(self.conversation_id, limit)
    
    def get_chatbot_format_history(self, history=None):
        """
        Format the conversation history for Streamlit chat display.
        
        Args:
            history (list, optional): List of message dictionaries from the database
            
        Returns:
            list: List of (role, content) tuples for Streamlit chat messages
        """
        if history is None:
            history = self.get_conversation_history()
        
        # Format for Streamlit chat
        formatted_messages = []
        
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "assistant"
                content = msg["content"]
                
                # Add the message content
                formatted_messages.append((role, content))
            
        return formatted_messages
        
    def get_agent_state(self):
        """
        Get the current internal state of the agent.
        
        Returns:
            dict: The agent's current memory, plans, and performance metrics
        """
        return {
            "memory": self.memory.get_memory_context(),
            "last_plan": self.last_plan,
            "last_query_type": self.last_query_type,
            "performance_metrics": self.performance_metrics,
            "model": self.selected_model,
            "model_version": self.model_version
        }
    
    def get_agent_memory_summary(self):
        """
        Get a human-readable summary of the agent's memory.
        
        Returns:
            str: Summary of the agent's memory and reflections
        """
        memory = self.memory.get_memory_context()
        
        summary = "Agent Memory Summary:\n\n"
        
        # Add reflections
        if memory["reflections"]:
            summary += "Recent Reflections:\n"
            for reflection in memory["reflections"][-3:]:
                summary += f"- {reflection}\n"
        
        # Add facts
        if memory["facts"]:
            summary += "\nDiscovered Facts:\n"
            for key, value in memory["facts"].items():
                summary += f"- {key}: {value}\n"
        
        # Add user preferences
        if memory["user_preferences"]:
            summary += "\nUser Preferences:\n"
            for pref_type, value in memory["user_preferences"].items():
                summary += f"- {pref_type}: {value}\n"
        
        # Add performance metrics summary
        summary += "\nPerformance Metrics:\n"
        for metric, value in self.performance_metrics.items():
            summary += f"- {metric}: {value}\n"
        
        return summary
    
    def reset_memory(self):
        """Reset the agent's memory and metrics."""
        self.memory = AgentMemory()
        self.performance_metrics = {
            "successful_responses": 0,
            "failed_responses": 0,
            "rag_hits": 0,
            "image_analyses": 0,
            "document_analyses": 0
        }
    
    def process_query(self, query, vector_store=None, image=None, document_content=None):
        """
        Process a user query with enhanced reasoning and agency capabilities.
        
        Args:
            query (str): The user's text query
            vector_store (tuple, optional): FAISS vector store for RAG
            image (Image, optional): PIL Image if an image was uploaded
            document_content (str, optional): Extracted text content from a document
            
        Returns:
            str: The AI's response
        """
        # Check for API key first
        if not self.api_key:
            api_key_missing_message = f"Please enter a valid API key for the {self.selected_model.capitalize()} model in the sidebar settings. You need an API key to interact with the AI models."
            # Save messages to DB for continuity
            add_message_to_db(self.conversation_id, "user", query)
            add_message_to_db(self.conversation_id, "assistant", api_key_missing_message)
            return api_key_missing_message
            
        # Save user message to DB
        add_message_to_db(self.conversation_id, "user", query)
        
        # Determine which model's functions to use
        if self.selected_model == "gemini":
            get_response = gemini_get_response
            analyze_image = gemini_analyze_image
            get_embedding = gemini_get_embedding
            get_rag_response = gemini_get_rag_response
            get_multimodal_response = gemini_get_multimodal_response
        else:  # OpenAI
            get_response = openai_get_response
            analyze_image = openai_analyze_image
            get_embedding = openai_get_embedding
            get_rag_response = openai_get_rag_response
            get_multimodal_response = openai_get_multimodal_response
        
        try:
            # Enhanced agentic processing
            # 1. Create context for reasoning
            history = self.get_conversation_history(limit=5)
            conversation_context = []
            for msg in history:
                conversation_context.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # 2. Generate reasoning plan
            logger.info(f"Creating reasoning plan for query: {query}")
            self.last_plan = self.reasoning.create_plan(query, conversation_context)
            logger.info(f"Execution plan: {self.last_plan}")
            
            # 3. Extract entities and concepts from query for enhanced processing
            entities = self.tools.extract_entities(query)
            concepts = self.tools.extract_concepts(query)
            logger.info(f"Extracted entities: {entities}, concepts: {concepts}")
            
            # 4. Execute the plan with error handling and fallbacks
            response_text = ""
            execution_notes = []
            
            # Determine query type and context
            if image:
                self.last_query_type = "image_query"
                execution_notes.append("Detected image input - using multimodal reasoning")
                
                try:
                    # Analyze image
                    image_analysis = analyze_image(image, api_key=self.api_key, model_version=self.model_version)
                    execution_notes.append("Image analysis completed successfully")
                    
                    # Get response that incorporates both query and image analysis
                    response_text = get_multimodal_response(query, image_analysis, api_key=self.api_key, model_version=self.model_version)
                    
                    # Store memory about the image
                    detected_objects = image_analysis.split('\n')[0]  # Simple extraction of detected objects
                    self.memory.add_fact("last_image_content", detected_objects)
                    self.performance_metrics["image_analyses"] += 1
                    
                except Exception as e:
                    logger.error(f"Error in image processing: {str(e)}")
                    execution_notes.append(f"Image analysis failed: {str(e)}")
                    
                    # Use fallback plan
                    fallback = self.reasoning.create_fallback_plan("analyze_image")
                    execution_notes.append(f"Using fallback plan: {fallback}")
                    self.last_fallback = fallback
                    
                    # Execute fallback - text-only response
                    response_text = get_response(
                        f"The following query was about an image, but I couldn't analyze it properly. "
                        f"Please respond helpfully anyway: {query}", 
                        api_key=self.api_key, 
                        model_version=self.model_version
                    )
            
            elif document_content:
                self.last_query_type = "document_query"
                execution_notes.append("Detected document input - using document analysis reasoning")
                
                try:
                    # Create a summary of the document
                    doc_summary = get_document_summary(document_content)
                    execution_notes.append("Document summary created successfully")
                    
                    # Extract key concepts from document for memory
                    doc_concepts = self.tools.extract_concepts(doc_summary)
                    self.memory.add_fact("document_concepts", doc_concepts)
                    
                    # Prepare a context for RAG with the document content
                    context = [{
                        "content": f"Document Analysis: {doc_summary}\n\n{document_content[:5000]}"  # Limit to first 5000 chars
                    }]
                    
                    # Get RAG-enhanced response
                    response_text = get_rag_response(query, context, api_key=self.api_key, model_version=self.model_version)
                    self.performance_metrics["document_analyses"] += 1
                    
                except Exception as e:
                    logger.error(f"Error in document processing: {str(e)}")
                    execution_notes.append(f"Document analysis failed: {str(e)}")
                    
                    # Use fallback plan
                    fallback = self.reasoning.create_fallback_plan("extract_document_info")
                    execution_notes.append(f"Using fallback plan: {fallback}")
                    self.last_fallback = fallback
                    
                    # Fallback to basic response
                    response_text = get_response(
                        f"I was asked to analyze a document with this query: {query}. "
                        f"However, I encountered an error processing the document. "
                        f"Could you please provide more information or try a different document format?",
                        api_key=self.api_key,
                        model_version=self.model_version
                    )
            
            elif vector_store:
                self.last_query_type = "knowledge_query"
                execution_notes.append("Using knowledge base for enhanced response")
                
                try:
                    # First, try generating alternative search queries for better RAG results
                    alt_queries = self.tools.generate_search_queries(query)
                    execution_notes.append(f"Generated alternative search queries: {alt_queries}")
                    
                    # Get embedding for the primary query
                    query_embedding = get_embedding(query, api_key=self.api_key, model_version=self.model_version)
                    
                    # Search vector store for relevant context
                    search_results = search_vector_store(vector_store, query_embedding, k=3)
                    
                    if search_results:
                        execution_notes.append(f"Found {len(search_results)} relevant knowledge items")
                        
                        # Get RAG-enhanced response with context
                        response_text = get_rag_response(query, search_results, api_key=self.api_key, model_version=self.model_version)
                        self.performance_metrics["rag_hits"] += 1
                        
                        # Store memory about successful knowledge retrieval
                        self.memory.add_fact("last_successful_knowledge_query", query)
                    else:
                        # Try alternative queries for better results
                        execution_notes.append("No direct matches found, trying alternative queries")
                        
                        for alt_query in alt_queries:
                            alt_embedding = get_embedding(alt_query, api_key=self.api_key, model_version=self.model_version)
                            alt_results = search_vector_store(vector_store, alt_embedding, k=2)
                            
                            if alt_results:
                                execution_notes.append(f"Found results with alternative query: {alt_query}")
                                search_results = alt_results
                                response_text = get_rag_response(
                                    query, 
                                    search_results, 
                                    api_key=self.api_key, 
                                    model_version=self.model_version
                                )
                                self.performance_metrics["rag_hits"] += 1
                                break
                                
                        if not response_text:
                            # Fallback to regular response if no context found with any query
                            execution_notes.append("No knowledge matches found, using direct response")
                            response_text = get_response(query, api_key=self.api_key, model_version=self.model_version)
                    
                except Exception as e:
                    logger.error(f"Error in RAG query processing: {e}")
                    execution_notes.append(f"Knowledge search failed: {str(e)}")
                    
                    # Fallback to regular response
                    response_text = get_response(query, api_key=self.api_key, model_version=self.model_version)
            
            else:
                # Regular text query
                self.last_query_type = "conversation"
                execution_notes.append("Using direct conversation reasoning")
                
                # Add memory to system prompt
                memory_context = ""
                recent_reflections = self.memory.get_recent_reflections()
                
                if recent_reflections:
                    memory_context = "Previous insights about this conversation:\n"
                    memory_context += "\n".join([f"- {r}" for r in recent_reflections])
                
                # Use system prompt with memory if available
                system_prompt = memory_context if memory_context else None
                response_text = get_response(
                    query, 
                    system_prompt=system_prompt,
                    api_key=self.api_key, 
                    model_version=self.model_version
                )
            
            # 5. Evaluate and reflect on the response
            logger.info("Evaluating response quality")
            quality_score, quality_reason = self.reasoning.evaluate_response_quality(response_text, query)
            
            if quality_score > 0.7:
                self.performance_metrics["successful_responses"] += 1
                self.memory.add_reflection(f"Successfully answered a question about {', '.join(concepts)}")
            else:
                self.performance_metrics["failed_responses"] += 1
                self.memory.add_reflection(f"Response may need improvement: {quality_reason}")
            
            # 6. Update agent memory with insights
            if entities:
                for entity in entities:
                    self.memory.add_fact(f"mentioned_{entity.lower()}", True)
            
            # Log the execution
            logger.info(f"Query execution complete. Notes: {execution_notes}")
            logger.info(f"Memory state: {self.memory.get_memory_context()}")
            logger.info(f"Performance metrics: {self.performance_metrics}")
            
            # Save assistant's response to DB
            message_id = add_message_to_db(self.conversation_id, "assistant", response_text)
            self.last_message_id = message_id
            
            return response_text
            
        except Exception as e:
            logger.error(f"Critical error in query processing: {str(e)}")
            error_message = f"Error: {str(e)}"
            # Save error message as assistant response
            add_message_to_db(self.conversation_id, "assistant", error_message)
            self.performance_metrics["failed_responses"] += 1
            return error_message
