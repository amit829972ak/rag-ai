"""
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
