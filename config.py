"""Configuration settings for the web search and report generation system."""

from agno.models.groq import Groq

# API Key for NVIDIA API
GROQ_API_KEY = "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"

DEFAULT_TOPIC = "Is there a process of establishment of Israeli Military or Offensive Cyber Industry in Australia?"

# Initial websites for crawling
INITIAL_WEBSITES = [
    "https://www.bellingcat.com/",
    "https://worldview.stratfor.com/",
    "https://thesoufancenter.org/",
    "https://www.globalsecurity.org/",
    "https://www.defenseone.com/"
]

# Model configuration
SEARCHER_MODEL_CONFIG = {
    "id": "llama-3.3-70b-versatile",
    "temperature": 0.4,
    "top_p": 0.3
}

# Model configuration
WRITER_MODEL_CONFIG = {
    "id": "llama-3.3-70b-versatile",
    "temperature": 0.2,
    "top_p": 0.2
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
