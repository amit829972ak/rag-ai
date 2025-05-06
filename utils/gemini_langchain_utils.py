from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

from utils.gemini_utils import get_ai_response

def create_rag_chain(api_key=None, model_version=None):
    """
    Create a RAG chain using LangChain with Google Gemini.
    
    Args:
        api_key (str, optional): Google Gemini API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        LLMChain: A LangChain chain for RAG responses.
    """
    # Create a PromptTemplate with placeholders
    template = """
    You are a helpful assistant that provides accurate information based on the context provided.
    
    Context information:
    {context}
    
    User question: {question}
    
    Please answer the question based on the context information. If the context doesn't 
    contain relevant information, acknowledge this and provide a general response.
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    # Get the appropriate model name from the model version
    from utils.gemini_utils import GEMINI_MODELS, DEFAULT_TEXT_MODEL
    
    # Use the specified model or the default
    selected_model = model_version if model_version and model_version in GEMINI_MODELS else DEFAULT_TEXT_MODEL
    
    # Get the API model name
    model_name = GEMINI_MODELS[selected_model]["api_name"]
    
    # Create the LLM with Google Gemini
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3
    )
    
    # Create the chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False
    )
    
    return chain

def get_rag_response(query, context, api_key=None, model_version=None):
    """
    Get a RAG-enhanced response using the provided context.
    
    Args:
        query (str): The user's query.
        context (list): List of relevant information from the vector store.
        api_key (str, optional): Google Gemini API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        str: The RAG-enhanced response.
    """
    try:
        # Create the chain
        chain = create_rag_chain(api_key, model_version)
        
        # Convert the context list to a string
        context_str = "\n\n".join(context)
        
        # Run the chain
        response = chain.run({"context": context_str, "question": query})
        return response
    except Exception as e:
        # Fallback to regular response
        system_prompt = f"""
        You are an assistant that answers questions based on the following context:
        
        {context}
        """
        return get_ai_response(query, system_prompt=system_prompt, api_key=api_key, model_version=model_version)

def get_multimodal_response(query, image_analysis, api_key=None, model_version=None):
    """
    Get a response that incorporates both text query and image analysis.
    
    Args:
        query (str): The user's text query.
        image_analysis (str): Analysis of the uploaded image.
        api_key (str, optional): Google Gemini API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        str: Response that considers both the text and image.
    """
    # For Google Gemini, we can construct a prompt that includes both
    system_prompt = f"""
    You are an assistant that answers questions based on both text and image content.
    
    Image analysis:
    {image_analysis}
    
    Please answer the user's question in relation to this image.
    """
    
    return get_ai_response(query, system_prompt=system_prompt, api_key=api_key, model_version=model_version)
