"""Configuration settings for the web search and report generation system."""

from phi.model.groq import Groq
from phi.model.together import Together
from phi.model.huggingface import HuggingFaceChat

# API Key for NVIDIA API
NVIDIA_API_KEY = "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"
GROQ_API_KEY = "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"
TOGETHER_API_KEY = "2cbe6e8b3620661ae7a9a963998e9d0d7a9122f486646056adbf57eff49080e4"
HF_API_KEY = "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"

# DEFAULT_TOPIC = "Is there a process of establishment of Israeli Military or Offensive Cyber Industry in Australia?"

# # Initial websites for crawling
# INITIAL_WEBSITES = [
#     "https://www.bellingcat.com/",
#     "https://worldview.stratfor.com/",
#     "https://thesoufancenter.org/",
#     "https://www.globalsecurity.org/",
#     "https://www.defenseone.com/"
# ]

# Model configuration
SEARCHER_MODEL_CONFIG = {
    "id": "Trelis/Meta-Llama-3-70B-Instruct-function-calling",
    "temperature": 0.4,
    "top_p": 0.3,
    "repetition_penalty": 1
}

# Model configuration
WRITER_MODEL_CONFIG = {
    "id": "Trelis/Meta-Llama-3-70B-Instruct-function-calling",
    "temperature": 0.2,
    "top_p": 0.2,
    "repetition_penalty": 1
}

# Review criteria thresholds
REVIEW_THRESHOLDS = {
    "min_word_count": 2000,
    "min_score": 7,
    "min_avg_score": 8,
    "max_iterations": 5
}

# Crawler settings
CRAWLER_CONFIG = {
    "max_pages_per_site": 10,
    "min_relevance_score": 0.5
}

def get_hf_model(purpose: str) -> HuggingFaceChat:
    """
    Factory function to create HuggingFaceChat models with specific configurations.
    
    Args:
        purpose: Either 'searcher' or 'writer' to determine which configuration to use
        
    Returns:
        Configured HuggingFaceChat model instance
    """
    if purpose == 'searcher':
        return HuggingFaceChat(
            id=SEARCHER_MODEL_CONFIG["id"],
            api_key=HF_API_KEY,
            temperature=SEARCHER_MODEL_CONFIG["temperature"],
            top_p=SEARCHER_MODEL_CONFIG["top_p"],
        )
    elif purpose == 'writer':
        return HuggingFaceChat(
            id=WRITER_MODEL_CONFIG["id"],
            api_key=HF_API_KEY,
            temperature=WRITER_MODEL_CONFIG["temperature"],
            top_p=WRITER_MODEL_CONFIG["top_p"]
        )
    else:
        raise ValueError(f"Unknown purpose: {purpose}. Must be 'searcher' or 'writer'")

def get_together_model(purpose: str) -> Together:
    """
    Factory function to create Together models with specific configurations.
    
    Args:
        purpose: Either 'searcher' or 'writer' to determine which configuration to use
        
    Returns:
        Configured Together model instance
    """
    if purpose == 'searcher':
        return Together(
            id=SEARCHER_MODEL_CONFIG["id"],
            api_key=TOGETHER_API_KEY,
            temperature=SEARCHER_MODEL_CONFIG["temperature"],
            top_p=SEARCHER_MODEL_CONFIG["top_p"],
            repetition_penalty=SEARCHER_MODEL_CONFIG["repetition_penalty"]
        )
    elif purpose == 'writer':
        return Together(
            id=WRITER_MODEL_CONFIG["id"],
            api_key=TOGETHER_API_KEY,
            temperature=WRITER_MODEL_CONFIG["temperature"],
            top_p=WRITER_MODEL_CONFIG["top_p"],
            repetition_penalty=WRITER_MODEL_CONFIG["repetition_penalty"]
        )
    else:
        raise ValueError(f"Unknown purpose: {purpose}. Must be 'searcher' or 'writer'")


def get_groq_model(purpose: str) -> Groq:
    """
    Factory function to create Groq models with specific configurations.
    
    Args:
        purpose: Either 'searcher' or 'writer' to determine which configuration to use
        
    Returns:
        Configured Groq model instance
    """
    if purpose == 'searcher':
        return Groq(
            id=SEARCHER_MODEL_CONFIG["id"],
            api_key=GROQ_API_KEY,
            temperature=SEARCHER_MODEL_CONFIG["temperature"],
            top_p=SEARCHER_MODEL_CONFIG["top_p"]
        )
    elif purpose == 'writer':
        return Groq(
            id=WRITER_MODEL_CONFIG["id"],
            api_key=GROQ_API_KEY,
            temperature=WRITER_MODEL_CONFIG["temperature"],
            top_p=WRITER_MODEL_CONFIG["top_p"]
        )
    else:
        raise ValueError(f"Unknown purpose: {purpose}. Must be 'searcher' or 'writer'")
