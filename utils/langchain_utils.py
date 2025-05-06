from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from utils.openai_utils import get_ai_response

def create_rag_chain(api_key=None, model_version=None):
    """
    Create a RAG chain using LangChain.
    
    Args:"""
Agent Reasoning Module - Enhances the agent with planning and reasoning capabilities.
"""
import json
import re
from typing import List, Dict, Any, Optional, Tuple


class AgentMemory:
    """
    Manages the agent's memory including conversation memory and reflections.
    """
    def __init__(self):
        self.reflections = []
        self.facts = {}
        self.user_preferences = {}
    
    def add_reflection(self, reflection: str):
        """Add a reflection about the conversation."""
        self.reflections.append(reflection)
        
    def add_fact(self, key: str, value: Any):
        """Store a discovered fact about the user or conversation."""
        self.facts[key] = value
        
    def add_user_preference(self, preference_type: str, value: str):
        """Store user preferences for future reference."""
        self.user_preferences[preference_type] = value
        
    def get_recent_reflections(self, count: int = 3) -> List[str]:
        """Get the most recent reflections."""
        return self.reflections[-count:] if self.reflections else []
    
    def get_memory_context(self) -> Dict[str, Any]:
        """Get the full memory context for the agent."""
        return {
            "reflections": self.reflections,
            "facts": self.facts,
            "user_preferences": self.user_preferences
        }


class ReasoningEngine:
    """
    Provides reasoning and planning capabilities for the agent.
    """
    def __init__(self):
        self.current_plan = []
        self.fallback_plans = {}
        
    def create_plan(self, query: str, context: List[Dict[str, str]]) -> List[str]:
        """
        Create a plan based on the query and available context.
        
        Args:
            query: The user's query
            context: Additional context information
            
        Returns:
            A list of steps to execute
        """
        # Analyze the query to determine intent
        intent = self._determine_intent(query)
        
        if "image" in intent:
            return ["analyze_image", "extract_key_elements", "relate_to_query", "formulate_response"]
        elif "document" in intent:
            return ["extract_document_info", "identify_key_points", "relate_to_query", "formulate_response"]
        elif "knowledge" in intent:
            return ["search_knowledge_base", "evaluate_relevance", "synthesize_information", "formulate_response"]
        else:
            return ["process_query", "generate_direct_response"]
    
    def _determine_intent(self, query: str) -> List[str]:
        """Determine the intent of the query."""
        intents = []
        
        # Simple keyword-based intent recognition
        if re.search(r'\b(image|picture|photo|see|look|visually)\b', query, re.IGNORECASE):
            intents.append("image")
            
        if re.search(r'\b(document|file|text|read|pdf|doc|docx)\b', query, re.IGNORECASE):
            intents.append("document")
            
        if re.search(r'\b(know|learn|understand|explain|what is|how does|why is)\b', query, re.IGNORECASE):
            intents.append("knowledge")
            
        if not intents:
            intents.append("conversation")
            
        return intents
    
    def create_fallback_plan(self, step: str) -> List[str]:
        """Create a fallback plan if a specific step fails."""
        fallbacks = {
            "analyze_image": ["describe_general_features", "use_text_only_response"],
            "search_knowledge_base": ["use_general_knowledge", "ask_for_clarification"],
            "extract_document_info": ["focus_on_available_content", "request_more_context"]
        }
        
        return fallbacks.get(step, ["generate_basic_response"])
    
    def evaluate_response_quality(self, response: str, query: str) -> Tuple[float, str]:
        """
        Evaluate the quality of a response.
        
        Args:
            response: The response to evaluate
            query: The original query
            
        Returns:
            A tuple of (score, reason)
        """
        score = 0.0
        reason = ""
        
        # Check response length
        if len(response) < 50:
            score -= 0.2
            reason += "Response is too short. "
        elif len(response) > 500:
            score += 0.1
            reason += "Response is detailed. "
            
        # Check if response directly addresses the query
        query_keywords = set(re.findall(r'\b\w{3,}\b', query.lower()))
        response_keywords = set(re.findall(r'\b\w{3,}\b', response.lower()))
        
        keyword_overlap = len(query_keywords.intersection(response_keywords)) / max(len(query_keywords), 1)
        
        if keyword_overlap > 0.5:
            score += 0.3
            reason += "Response addresses query keywords well. "
        else:
            score -= 0.2
            reason += "Response may not directly address the query. "
            
        # Check for uncertainty markers
        uncertainty_phrases = ["I'm not sure", "I don't know", "cannot determine", "unclear"]
        if any(phrase in response for phrase in uncertainty_phrases):
            score -= 0.1
            reason += "Response contains uncertainty. "
            
        # Normalize score to 0-1 range
        score = max(0.0, min(1.0, score + 0.5))  # Default middle score is 0.5
        
        return (score, reason)


class AgentTools:
    """
    Provides utility tools for the agent.
    """
    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """Extract potential entities from text."""
        # Simple regex-based entity extraction
        entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        return list(set(entities))
    
    @staticmethod
    def extract_concepts(text: str) -> List[str]:
        """Extract key concepts from text."""
        # Extract noun phrases or important keywords
        words = text.lower().split()
        # Remove stop words and keep words longer than 3 characters
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like'}
        concepts = [word for word in words if word not in stop_words and len(word) > 3]
        return list(set(concepts))
    
    @staticmethod
    def generate_search_queries(text: str, num_queries: int = 3) -> List[str]:
        """Generate alternative search queries based on the text."""
        entities = AgentTools.extract_entities(text)
        concepts = AgentTools.extract_concepts(text)
        
        queries = []
        if entities and concepts:
            for i in range(min(num_queries, len(entities))):
                entity = entities[i % len(entities)]
                concept = concepts[i % len(concepts)]
                queries.append(f"{entity} {concept}")
                
        if len(queries) < num_queries:
            # Generate additional queries from the text
            words = text.split()
            for i in range(len(queries), num_queries):
                if len(words) >= 3:
                    start_idx = (i * 2) % (len(words) - 2)
                    queries.append(" ".join(words[start_idx:start_idx+3]))
                else:
                    queries.append(text)
                    
        return queries[:num_queries]
        api_key (str, optional): OpenAI API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        LLMChain: A LangChain chain for RAG responses.
    """
    # Create a custom prompt template for RAG
    rag_template = """
    Answer the query based on the provided context.
    
    Context:
    {context}
    
    Query: {query}
    
    Answer:
    """
    
    # Initialize the prompt with the template
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=rag_template
    )
    
    # Initialize the language model
    try:
        # Get the appropriate model name from the model version
        from utils.openai_utils import OPENAI_MODELS, DEFAULT_MODEL
        
        # Use the specified model or the default
        selected_model = model_version if model_version and model_version in OPENAI_MODELS else DEFAULT_MODEL
        
        # Get the API model name
        model_name = OPENAI_MODELS[selected_model]["api_name"]
        
        # Use the provided API key if available
        if api_key:
            llm = ChatOpenAI(temperature=0.7, max_tokens=800, openai_api_key=api_key, model=model_name)
        else:
            llm = ChatOpenAI(temperature=0.7, max_tokens=800, model=model_name)
        
        # Create the chain
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=False
        )
        
        return chain
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        return None

def get_rag_response(query, context, api_key=None, model_version=None):
    """
    Get a RAG-enhanced response using the provided context.
    
    Args:
        query (str): The user's query.
        context (list): List of relevant information from the vector store.
        api_key (str, optional): OpenAI API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        str: The RAG-enhanced response.
    """
    try:
        # Create the chain with API key and model version
        chain = create_rag_chain(api_key=api_key, model_version=model_version)
        
        if chain:
            # Format context as a string
            context_str = "\n\n".join([f"- {item['content']}" for item in context])
            
            # Get response from chain
            response = chain.run(context=context_str, query=query)
            return response
        else:
            # Fallback to direct API call
            context_str = "\n\n".join([f"- {item['content']}" for item in context])
            system_prompt = f"""
            You are a helpful assistant. Answer the user's question based on this context:
            
            {context_str}
            
            Only use information from the context to answer. If the context doesn't contain 
            relevant information, acknowledge this and provide a general response.
            """
            
            return get_ai_response(query, system_prompt=system_prompt, api_key=api_key, model_version=model_version)
    except Exception as e:
        print(f"Error in RAG response: {e}")
        return f"I encountered an issue while retrieving information: {str(e)}. Let me try to answer more generally."

def get_multimodal_response(query, image_analysis, api_key=None, model_version=None):
    """
    Get a response that incorporates both text query and image analysis.
    
    Args:
        query (str): The user's text query.
        image_analysis (str): Analysis of the uploaded image.
        api_key (str, optional): OpenAI API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        str: Response that considers both the text and image.
    """
    system_prompt = f"""
    You are a helpful assistant with the ability to see images. A user has uploaded an image 
    and asked a question or made a comment about it.
    
    The following is an analysis of the image content:
    {image_analysis}
    
    Based on this image analysis and the user's question, provide a thoughtful and relevant response.
    """
    
    return get_ai_response(query, system_prompt=system_prompt, api_key=api_key, model_version=model_version)
