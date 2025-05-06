import os
import io
import base64
from PIL import Image
import google.generativeai as genai

# Gemini model options with expanded set of models
GEMINI_MODELS = {
    "gemini-1.0-pro": {
        "name": "Gemini 1.0 Pro", 
        "description": "Free tier capable model, good balance of capability and cost",
        "api_name": "gemini-pro"
    },
    "gemini-1.0-pro-vision": {
        "name": "Gemini 1.0 Pro Vision", 
        "description": "Free tier model with image understanding",
        "api_name": "gemini-pro-vision"
    },
    "gemini-1.5-pro": {
        "name": "Gemini 1.5 Pro", 
        "description": "Advanced model with improved capabilities",
        "api_name": "gemini-1.5-pro"
    },
    "gemini-1.5-pro-latest": {
        "name": "Gemini 1.5 Pro (Latest)", 
        "description": "Latest version with the most recent improvements",
        "api_name": "gemini-1.5-pro-latest"
    },
    "gemini-1.5-flash": {
        "name": "Gemini 1.5 Flash", 
        "description": "Faster model, good for quick responses",
        "api_name": "gemini-1.5-flash"
    },
    "gemini-1.5-flash-latest": {
        "name": "Gemini 1.5 Flash (Latest)", 
        "description": "Latest version of the faster model",
        "api_name": "gemini-1.5-flash-latest"
    },
    "gemini-2.0-pro": {
        "name": "Gemini 2.0 Pro", 
        "description": "Next-generation model with enhanced reasoning",
        "api_name": "gemini-2.0-pro"
    },
    "gemini-2.0-pro-vision": {
        "name": "Gemini 2.0 Pro Vision", 
        "description": "Advanced vision capabilities with improved understanding",
        "api_name": "gemini-2.0-pro-vision"
    },
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro", 
        "description": "Latest model with state-of-the-art capabilities",
        "api_name": "gemini-2.5-pro"
    },
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash", 
        "description": "Fastest next-gen model for efficient responses",
        "api_name": "gemini-2.5-flash"
    }
}

# Models for specific capabilities
GEMINI_VISION_MODELS = [
    "gemini-1.0-pro-vision", 
    "gemini-1.5-pro", 
    "gemini-1.5-flash", 
    "gemini-1.5-pro-latest", 
    "gemini-1.5-flash-latest",
    "gemini-2.0-pro-vision",
    "gemini-2.0-pro",
    "gemini-2.5-pro",
    "gemini-2.5-flash"
]
GEMINI_EMBEDDING_MODEL = "models/embedding-001"

# Default models
DEFAULT_TEXT_MODEL = "gemini-1.0-pro"  # For text-only queries (free tier capable)
DEFAULT_VISION_MODEL = "gemini-1.0-pro-vision"  # For queries with images (free tier capable)

def get_gemini_client(api_key=None):
    """
    Configure the Gemini API with the provided API key.
    
    Args:
        api_key (str, optional): Google Gemini API key
        
    Returns:
        None: Configuration is set globally
        
    Raises:
        Exception: If API key is invalid or missing
    """
    # Require API key - no fallback to environment variable
    if not api_key:
        raise Exception("No API key provided. Please provide a Google Gemini API key.")
    
    key = api_key
    
    # Configure the Gemini API but avoid test API calls during validation
    try:
        # Only configure the API without making test calls
        genai.configure(api_key=key)
        
        # Simple format validation (API keys are usually ~40+ characters)
        if len(key) < 20:
            raise Exception("API key appears to be too short. Google Gemini API keys are typically longer.")
        
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid" in error_msg and "api" in error_msg:
            raise Exception("Invalid API key. Please check your Google Gemini API key and try again.")
        elif "rate" in error_msg or "quota" in error_msg or "limit" in error_msg:
            raise Exception("API rate limit reached. Please try again later or use a different API key.")
        else:
            raise Exception(f"Error configuring Gemini API: {str(e)}")
    
def get_ai_response(prompt, system_prompt=None, context=None, api_key=None, model_version=None):
    """
    Get a response from the Google Gemini API.
    
    Args:
        prompt (str): The user's prompt.
        system_prompt (str, optional): System instructions to guide the AI.
        context (list, optional): Previous conversation context.
        api_key (str, optional): Google Gemini API key.
        model_version (str, optional): The specific Gemini model to use.
        
    Returns:
        str: The AI's response.
        
    Raises:
        Exception: If there's an API error
    """
    try:
        # Configure the client
        get_gemini_client(api_key)
        
        # Use the specified model or the default
        selected_model = model_version if model_version and model_version in GEMINI_MODELS else DEFAULT_TEXT_MODEL
        
        # Get the API name for the selected model
        model_name = GEMINI_MODELS[selected_model]["api_name"]
        
        # Create the model
        model = genai.GenerativeModel(model_name)
        
        # Prepare conversation history if available
        chat_history = []
        if context:
            for message in context:
                role = "user" if message["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": [message["content"]]})
        
        # Start a chat session
        chat = model.start_chat(history=chat_history if chat_history else None)
        
        # Add system prompt as an initial instruction if provided
        if system_prompt:
            # For Gemini, system prompts are part of the user message
            prompt = f"{system_prompt}\n\nUser query: {prompt}"
        
        # Send the message and get the response
        response = chat.send_message(prompt)
        
        return response.text
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid Google API key. Please check your API key settings and try again.")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise Exception(f"Google API quota or rate limit reached: {error_msg}")
        else:
            raise Exception(f"Error generating response: {error_msg}")

def encode_image_to_base64(image):
    """
    Encode an image to base64 for API transmission.
    
    Args:
        image (PIL.Image): The image to encode.
        
    Returns:
        str: Base64-encoded image string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_image_content(image, api_key=None, model_version=None):
    """
    Analyze an image using Google Gemini's vision capabilities.
    
    Args:
        image (PIL.Image): The image to analyze.
        api_key (str, optional): Google Gemini API key.
        model_version (str, optional): The specific Gemini model to use.
        
    Returns:
        str: Analysis of the image content.
    """
    try:
        # Configure the client
        get_gemini_client(api_key)
        
        # Use a vision-capable model
        # If the selected model supports vision, use it; otherwise use the default vision model
        selected_model = DEFAULT_VISION_MODEL  # Default vision-capable model
        
        # If a specific model was requested and it's in our vision models list, use it
        if model_version and model_version in GEMINI_VISION_MODELS:
            selected_model = model_version
        
        # Ensure we have a string value
        selected_model = selected_model if selected_model else DEFAULT_VISION_MODEL
        
        # Get the API name for the selected model
        if selected_model in GEMINI_MODELS:
            vision_model = GEMINI_MODELS[selected_model]["api_name"]
        else:
            # Fallback to the default vision model's API name
            vision_model = "gemini-pro-vision"
            
        # Create the vision model
        model = genai.GenerativeModel(vision_model)
        
        # Convert image to bytes for Gemini
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Create a structured prompt with image
        prompt = "Analyze this image in detail. Describe what you see and provide any relevant information about the content."
        
        # Send the prompt with the image
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
        
        return response.text
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid Google API key. Please check your API key settings and try again.")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise Exception(f"Google API quota or rate limit reached: {error_msg}")
        elif "content" in error_msg.lower() and ("policy" in error_msg.lower() or "filter" in error_msg.lower()):
            raise Exception("The image appears to contain content that violates Google's content policy. Please try a different image.")
        else:
            raise Exception(f"Error analyzing image: {error_msg}")

def get_embedding(text, api_key=None, model_version=None):
    """
    Get an embedding vector for the given text using Google's embedding model.
    
    Args:
        text (str): The text to embed.
        api_key (str, optional): Google API key.
        model_version (str, optional): Not used for embeddings, but included for API consistency.
        
    Returns:
        list: The embedding vector.
    """
    try:
        # Configure the client
        get_gemini_client(api_key)
        
        # Create the embedding model
        embedding_model = "models/embedding-001"
        result = genai.embed_content(
            model=embedding_model,
            content=text,
            task_type="retrieval_query"
        )
        
        # Return the embedding values
        return result["embedding"]
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid Google API key. Please check your API key settings and try again.")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise Exception(f"Google API embedding generation rate limit reached: {error_msg}")
        else:
            raise Exception(f"Error generating embedding: {error_msg}")
