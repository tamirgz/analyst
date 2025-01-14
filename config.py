"""Configuration settings for the web search and report generation system."""

# API Key for NVIDIA API
NVIDIA_API_KEY = "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"
GROQ_API_KEY = "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"

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
