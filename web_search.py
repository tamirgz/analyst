import json
import re
import time
import os
import concurrent.futures
from typing import Optional, Iterator, List, Set, Dict, Any
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from datetime import datetime

# Phi imports
from agno.workflow import Workflow, RunResponse, RunEvent
from agno.storage.workflow.sqlite import SqliteWorkflowStorage
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.utils.pprint import pprint_run_response
from agno.utils.log import logger

import logging

# Error handling imports
from duckduckgo_search.exceptions import RatelimitException
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from requests.exceptions import HTTPError

from config import DEFAULT_TOPIC, INITIAL_WEBSITES, SEARCHER_MODEL_CONFIG, WRITER_MODEL_CONFIG, NVIDIA_API_KEY

DUCK_DUCK_GO_FIXED_MAX_RESULTS = 10

# Add at the top of the file after the logger import
logger.setLevel(logging.DEBUG)

# The topic to generate a blog post on
topic = DEFAULT_TOPIC

class NewsArticle(BaseModel):
    """Article data model containing title, URL and description."""
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    description: Optional[str] = Field(None, description="Summary of the article if available.")

class SearchResults(BaseModel):
    """Container for search results containing a list of articles."""
    articles: List[NewsArticle]

class BlogPostGenerator(Workflow):
    """Workflow for generating blog posts based on web research."""
    searcher: Agent = Field(...)
    backup_searcher: Agent = Field(...)
    writer: Agent = Field(...)
    initial_websites: List[str] = Field(default_factory=list)  # Changed this line
    file_handler: Optional[Any] = Field(None)

    def __init__(
        self,
        session_id: str,
        searcher: Agent,
        backup_searcher: Agent,
        writer: Agent,
        file_handler: Optional[Any] = None,
        storage: Optional[SqliteWorkflowStorage] = None,
        initial_websites: Optional[List[str]] = None,  # Added this parameter
    ):
        super().__init__(
            session_id=session_id,
            storage=storage,
        )
        self.file_handler = file_handler
        self.searcher = searcher
        self.backup_searcher = backup_searcher
        self.writer = writer
        # Initialize initial_websites with the provided list or default
        self.initial_websites = initial_websites if initial_websites is not None else INITIAL_WEBSITES.copy()
        
        # Configure search instructions
        search_instructions = [
            "Given a topic, search for 20 articles and return the 15 most relevant articles.",
            "For each article, provide:",
            "- title: The article title",
            "- url: The article URL",
            "- description: A brief description or summary of the article",
            "Return the results in a structured format with these exact field names."
        ]
        
        # Primary searcher using DuckDuckGo
        self.searcher = Agent(
            model=Groq(
                id=SEARCHER_MODEL_CONFIG["id"],
                api_key=NVIDIA_API_KEY,
                temperature=SEARCHER_MODEL_CONFIG["temperature"],
                top_p=SEARCHER_MODEL_CONFIG["top_p"]
            ),
            tools=[DuckDuckGoTools(fixed_max_results=DUCK_DUCK_GO_FIXED_MAX_RESULTS)],
            instructions=search_instructions,
            # response_model=SearchResults
        )
        
        # Backup searcher using Google Search
        self.backup_searcher = Agent(
            model=Groq(
                id=SEARCHER_MODEL_CONFIG["id"],
                api_key=NVIDIA_API_KEY,
                temperature=SEARCHER_MODEL_CONFIG["temperature"],
                top_p=SEARCHER_MODEL_CONFIG["top_p"]
            ),
            tools=[GoogleSearchTools()],
            instructions=search_instructions,
            # response_model=SearchResults
        )

        # Writer agent configuration
        writer_instructions = [
            "You are a professional research analyst tasked with creating a comprehensive report on the given topic.",
            "The sources provided include both general web search results and specialized intelligence/security websites.",
            "Carefully analyze and cross-reference information from all sources to create a detailed report.",
            "",
            "Report Structure:",
            "1. Executive Summary (2-3 paragraphs)",
            "   - Provide a clear, concise overview of the main findings",
            "   - Address the research question directly",
            "   - Highlight key discoveries and implications",
            "",
            "2. Detailed Analysis (Multiple sections)",
            "   - Break down the topic into relevant themes or aspects",
            "   - For each theme:",
            "     * Present detailed findings from multiple sources",
            "     * Cross-reference information between general and specialized sources",
            "     * Analyze trends, patterns, and developments",
            "     * Discuss implications and potential impacts",
            "",
            "3. Source Analysis and Credibility",
            "   For each major source:",
            "   - Evaluate source credibility and expertise",
            "   - Note if from specialized intelligence/security website",
            "   - Assess potential biases or limitations",
            "   - Key findings and unique contributions",
            "",
            "4. Key Takeaways and Strategic Implications",
            "   - Synthesize findings from all sources",
            "   - Compare/contrast general media vs specialized analysis",
            "   - Discuss broader geopolitical implications",
            "   - Address potential future developments",
            "",
            "5. References",
            "   - Group sources by type (specialized websites vs general media)",
            "   - List all sources with full citations",
            "   - Include URLs as clickable markdown links [Title](URL)",
            "   - Ensure every major claim has at least one linked source",
            "",
            "Important Guidelines:",
            "- Prioritize information from specialized intelligence/security sources",
            "- Cross-validate claims between multiple sources when possible",
            "- Maintain a professional, analytical tone",
            "- Support all claims with evidence",
            "- Include specific examples and data points",
            "- Use direct quotes for significant statements",
            "- Address potential biases in reporting",
            "- Ensure the report directly answers the research question",
            "",
            "Format the report with clear markdown headings (# ## ###), subheadings, and paragraphs.",
            "Each major section should contain multiple paragraphs with detailed analysis."
        ]
        
        self.writer = Agent(
            model=Groq(
                id=WRITER_MODEL_CONFIG["id"],
                api_key=NVIDIA_API_KEY,
                temperature=WRITER_MODEL_CONFIG["temperature"],
                top_p=WRITER_MODEL_CONFIG["top_p"]
            ),
            instructions=writer_instructions,
            structured_outputs=True
        )

    def _parse_search_response(self, response) -> Optional[SearchResults]:
        """Parse and validate search response into SearchResults model."""
        try:
            # Add debug logging for initial response
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Raw response content: {response[:500]}..." if isinstance(response, str) else str(response)[:500])

            if isinstance(response, str):
                # Clean up markdown code blocks and extract JSON
                content = response.strip()
                
                # Add debug logging for content before processing
                logger.debug(f"Content before processing: {content[:500]}...")

                if '```' in content:
                    # Extract content between code block markers
                    matches = re.finditer(r'```(?:json)?\n(.*?)\n```', content, re.DOTALL)
                    json_blocks = [match.group(1).strip() for match in matches]
                    
                    # Try each JSON block until we find a valid one
                    for json_block in json_blocks:
                        try:
                            data = json.loads(json_block)
                            if isinstance(data, dict) and 'articles' in data:
                                logger.debug(f"Found valid JSON block: {json_block[:500]}...")
                                content = json_block
                                break
                        except json.JSONDecodeError:
                            continue
                    
                    if not json_blocks:
                        # If no proper code block found, remove all ``` markers
                        content = re.sub(r'```(?:json)?\n?', '', content)
                        content = content.strip()
                
                # Add debug logging for content after markdown cleanup
                logger.debug(f"Content after markdown cleanup: {content[:500]}...")

                # Try to parse JSON response
                try:
                    # Clean up any trailing commas before closing brackets/braces
                    content = re.sub(r',(\s*[}\]])', r'\1', content)
                    # Fix invalid escape sequences
                    content = re.sub(r'\\([^"\\\/bfnrtu])', r'\1', content)
                    content = content.replace('\t', ' ')
                    # Handle any remaining unicode escapes
                    content = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), content)
                    
                    # Add debug logging for content after cleanup
                    logger.debug(f"Content after cleanup: {content[:500]}...")

                    # Try parsing with different JSON structures
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        # Try wrapping in articles array if not already
                        if not content.strip().startswith('{'):
                            content = f'{{"articles": {content}}}'
                            data = json.loads(content)
                    
                    if isinstance(data, dict) and 'articles' in data:
                        articles = []
                        for article in data['articles']:
                            if isinstance(article, dict):
                                article = {
                                    'title': str(article.get('title', '')).strip(),
                                    'url': str(article.get('url', '')).strip(),
                                    'description': str(article.get('description', '')).strip()
                                }
                                if article['title'] and article['url']:
                                    articles.append(NewsArticle(**article))
                        
                        if articles:
                            logger.info(f"Successfully parsed {len(articles)} articles from JSON")
                            return SearchResults(articles=articles)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {str(e)}\nContent: {content[:500]}...")
                    
                # Rest of the existing fallback code...
                # [Previous regex extraction code remains unchanged]

            elif isinstance(response, dict):
                # Add debug logging for dict response
                logger.debug(f"Processing dictionary response: {str(response)[:500]}...")
                # [Rest of dict handling remains unchanged]

            elif isinstance(response, RunResponse):
                # Add debug logging for RunResponse
                logger.debug(f"Processing RunResponse: {str(response.content)[:500] if response.content else 'No content'}")
                if response.content:
                    return self._parse_search_response(response.content)
                return None

            logger.error(f"Unsupported response type: {type(response)}")
            return None
                
        except Exception as e:
            logger.error(f"Error parsing search response: {str(e)}", exc_info=True)
            return None

    def _search_with_retry(self, topic: str, use_backup: bool = False, max_retries: int = 3) -> Optional[SearchResults]:
        """Execute search with retries and rate limit handling."""
        searcher = self.backup_searcher if use_backup else self.searcher
        source = "backup" if use_backup else "primary"
        
        # Initialize rate limit tracking
        rate_limited_sources = set()
        
        for attempt in range(max_retries):
            try:
                if source in rate_limited_sources:
                    logger.warning(f"{source} search is rate limited, switching to alternative method")
                    if not use_backup:
                        return self._search_with_retry(topic, use_backup=True, max_retries=max_retries)
                    backoff_time = min(3600, 60 * (2 ** attempt))
                    logger.info(f"All search methods rate limited. Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
                
                logger.info(f"\nAttempting {source} search (attempt {attempt + 1}/{max_retries})...")
                
                # Enhanced search prompt with explicit JSON structure
                search_prompt = f"""Search for detailed articles about: {topic}
                Return exactly 15 relevant articles in the following JSON format:
                {{
                    "articles": [
                        {{
                            "title": "Article Title",
                            "url": "https://example.com/article",
                            "description": "Brief description of the article"
                        }}
                    ]
                }}
                Ensure all JSON fields are properly quoted and formatted."""
                
                # Add timeout to prevent hanging
                try:
                    response = searcher.run(search_prompt, stream=False, timeout=30)
                except Exception as search_error:
                    logger.warning(f"Search error with {source}: {str(search_error)}")
                    if not use_backup:
                        logger.info("Switching to backup search method...")
                        return self._search_with_retry(topic, use_backup=True, max_retries=max_retries)
                    raise
                
                # Handle different response types
                if isinstance(response, RunResponse):
                    content = response.content
                elif isinstance(response, str):
                    content = response
                else:
                    logger.warning(f"Unexpected response type from {source}: {type(response)}")
                    if not use_backup:
                        return self._search_with_retry(topic, use_backup=True, max_retries=max_retries)
                    continue
                
                # Try to parse the response
                results = self._parse_search_response(content)
                if results and results.articles:
                    logger.info(f"Found {len(results.articles)} articles from {source} search")
                    return results
                
                logger.warning(f"{source.title()} search returned no valid results")
                if not use_backup:
                    return self._search_with_retry(topic, use_backup=True, max_retries=max_retries)
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(err in error_msg for err in ["rate", "limit", "quota", "exhausted", "none", "failed"]):
                    rate_limited_sources.add(source)
                    logger.error(f"{source} search rate limited: {str(e)}")
                    if not use_backup:
                        backup_results = self._search_with_retry(topic, use_backup=True, max_retries=max_retries)
                        if backup_results:
                            return backup_results
                else:
                    logger.error(f"Error during {source} search (attempt {attempt + 1}): {str(e)}")
                
                if attempt < max_retries - 1:
                    backoff_time = 2 ** attempt
                    if source in rate_limited_sources:
                        backoff_time = min(3600, 60 * (2 ** attempt))
                    logger.info(f"Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
        
        return None

    def _validate_content(self, content: str) -> bool:
        """Validate that the generated content is readable and properly formatted."""
        if not content or len(content.strip()) < 100:
            logger.warning("Content too short or empty")
            return False
            
        # Check for basic structure
        if not any(marker in content for marker in ['#', '\n\n']):
            logger.warning("Content lacks proper structure (headers or paragraphs)")
            return False
            
        # Check for reasonable paragraph lengths
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if not paragraphs:
            logger.warning("No valid paragraphs found")
            return False
            
        # Common words that are allowed to repeat frequently
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'it', 'its', 'is', 'are', 'was', 'were', 'be', 'been',
            'has', 'have', 'had', 'would', 'could', 'should', 'will', 'can'
        }
        
        # Track word frequencies across paragraphs
        word_frequencies = {}
        total_words = 0
        
        # Validate each paragraph
        for para in paragraphs:
            # Skip headers and references
            if para.startswith('#') or para.startswith('http'):
                continue
                
            # Calculate word statistics
            words = para.split()
            if len(words) < 3:
                continue  # Skip very short paragraphs
                
            # Calculate word statistics
            word_lengths = [len(word) for word in words]
            avg_word_length = sum(word_lengths) / len(word_lengths)
            
            # More nuanced word length validation
            long_words = [w for w in words if len(w) > 15]
            long_word_ratio = len(long_words) / len(words) if words else 0
            
            # Allow higher average length if the text contains URLs or technical terms
            contains_url = any(word.startswith(('http', 'www')) for word in words)
            contains_technical = any(word.lower().endswith(('tion', 'ment', 'ology', 'ware', 'tech')) for word in words)
            
            # Adjust thresholds based on content type
            max_avg_length = 12  # Base maximum average word length
            if contains_url:
                max_avg_length = 20  # Allow longer average for content with URLs
            elif contains_technical:
                max_avg_length = 15  # Allow longer average for technical content
            
            # Fail only if multiple indicators of problematic text
            if (avg_word_length > max_avg_length and long_word_ratio > 0.3) or avg_word_length > 25:
                logger.warning(f"Suspicious word lengths: avg={avg_word_length:.1f}, long_ratio={long_word_ratio:.1%}")
                return False
            
            # Check for excessive punctuation or special characters
            special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?()"-]', para)) / len(para)
            if special_char_ratio > 0.15:  # Increased threshold slightly
                logger.warning(f"Too many special characters: {special_char_ratio}")
                return False
                
            # Check for coherent sentence structure
            sentences = [s.strip() for s in re.split(r'[.!?]+', para) if s.strip()]
            weak_sentences = 0
            for sentence in sentences:
                words = sentence.split()
                if len(words) < 3:  # Skip very short sentences
                    continue
                    
                # More lenient grammar check
                structure_indicators = [
                    any(word[0].isupper() for word in words),  # Has some capitalization
                    any(word.lower() in common_words for word in words),  # Has common words
                    len(words) >= 3,  # Reasonable length
                    any(len(word) > 3 for word in words),  # Has some non-trivial words
                ]
                
                # Only fail if less than 2 indicators are present
                if sum(structure_indicators) < 2:
                    logger.warning(f"Weak sentence structure: {sentence}")
                    weak_sentences += 1
                    if weak_sentences > len(sentences) / 2:  # Fail if more than half are weak
                        logger.warning("Too many poorly structured sentences")
                        return False
                
                # Update word frequencies
                for word in words:
                    word = word.lower()
                    if word not in common_words and len(word) > 2:  # Only track non-common words
                        word_frequencies[word] = word_frequencies.get(word, 0) + 1
                        total_words += 1
        
        # Check for excessive repetition
        if total_words > 0:
            for word, count in word_frequencies.items():
                # Calculate the frequency as a percentage
                frequency = count / total_words
                
                # Allow up to 10% frequency for any word
                if frequency > 0.1 and count > 3:
                    logger.warning(f"Word '{word}' appears too frequently ({count} times, {frequency:.1%})")
                    return False
        
        # Content seems valid
        return True

    def _save_markdown(self, topic: str, content: str) -> str:
        """Save the content as an HTML file."""
        try:
            # Get or create report directory
            report_dir = None
            if hasattr(self, 'file_handler') and self.file_handler:
                report_dir = self.file_handler.report_dir
            else:
                # Create a default report directory if no file handler
                report_dir = os.path.join(os.path.dirname(__file__), f"report_{datetime.now().strftime('%Y-%m-%d')}")
                os.makedirs(report_dir, exist_ok=True)
                logger.info(f"Created report directory: {report_dir}")
            
            # Create filename from topic
            filename = re.sub(r'[^\w\s-]', '', topic.lower())  # Remove special chars
            filename = re.sub(r'[-\s]+', '-', filename)        # Replace spaces with hyphens
            filename = f"{filename}.html"
            file_path = os.path.join(report_dir, filename)
            
            # Convert markdown to HTML with styling
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{topic}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    h1 {{
                        color: #2c3e50;
                        border-bottom: 2px solid #3498db;
                        padding-bottom: 10px;
                    }}
                    h2 {{
                        color: #34495e;
                        margin-top: 30px;
                    }}
                    h3 {{
                        color: #455a64;
                    }}
                    a {{
                        color: #3498db;
                        text-decoration: none;
                    }}
                    a:hover {{
                        text-decoration: underline;
                    }}
                    .executive-summary {{
                        background-color: #f8f9fa;
                        border-left: 4px solid #3498db;
                        padding: 20px;
                        margin: 20px 0;
                    }}
                    .analysis-section {{
                        margin: 30px 0;
                    }}
                    .source-section {{
                        background-color: #f8f9fa;
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 5px;
                    }}
                    .references {{
                        margin-top: 40px;
                        border-top: 2px solid #ecf0f1;
                        padding-top: 20px;
                    }}
                    .timestamp {{
                        color: #7f8c8d;
                        font-size: 0.9em;
                        margin-top: 40px;
                        text-align: right;
                    }}
                    blockquote {{
                        border-left: 3px solid #3498db;
                        margin: 20px 0;
                        padding-left: 20px;
                        color: #555;
                    }}
                    code {{
                        background-color: #f7f9fa;
                        padding: 2px 5px;
                        border-radius: 3px;
                        font-family: monospace;
                    }}
                </style>
            </head>
            <body>
                <div class="content">
                    {self._markdown_to_html(content)}
                </div>
                <div class="timestamp">
                    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </body>
            </html>
            """
            
            # Write the HTML file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Successfully saved HTML report: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save HTML file: {str(e)}")
            return None
    
    def _markdown_to_html(self, markdown_content: str) -> str:
        """Convert markdown content to HTML with basic formatting."""
        # Headers
        html = markdown_content
        html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        
        # Lists
        html = re.sub(r'^\* (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'(<li>.*?</li>\n)+', r'<ul>\n\g<0></ul>', html, flags=re.DOTALL)
        
        # Links
        html = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', html)
        
        # Emphasis
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
        
        # Paragraphs
        html = re.sub(r'\n\n(.*?)\n\n', r'\n<p>\1</p>\n', html, flags=re.DOTALL)
        
        # Blockquotes
        html = re.sub(r'^\> (.*?)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
        
        # Code blocks
        html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
        html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)
        
        return html

    def run(self, topic: str, use_cache: bool = True) -> Iterator[RunResponse]:
        """Run the blog post generation workflow."""
        logger.info(f"Starting blog post generation for topic: {topic}")
        
        # Extract keywords from topic
        keywords = topic.lower().split()
        keywords = [w for w in keywords if len(w) > 3 and w not in {'what', 'where', 'when', 'how', 'why', 'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should', 'the', 'and', 'but', 'or', 'for', 'with'}]
        
        all_articles = []
        existing_urls = set()
        
        # First, try web search
        logger.info("Starting web search...")
        search_results = self._search_with_retry(topic)
        if search_results and search_results.articles:
            for article in search_results.articles:
                if article.url not in existing_urls:
                    all_articles.append(article)
                    existing_urls.add(article.url)
            logger.info(f"Found {len(search_results.articles)} articles from web search")
        
        # Then, crawl initial websites
        logger.info("Starting website crawl...")
        from file_handler import FileHandler
        crawler = WebsiteCrawler(max_pages_per_site=10)
        crawler.file_handler = FileHandler()  # Initialize file handler
        
        # Get the report directory from the file handler
        report_dir = crawler.file_handler.report_dir
        
        crawled_results = crawler.crawl_all_websites(self.initial_websites, keywords)
        
        # Save the relevance log to the report directory
        crawler.save_relevance_log(report_dir)
        
        if crawled_results:
            for result in crawled_results:
                if result['url'] not in existing_urls:
                    article = NewsArticle(**result)
                    all_articles.append(article)
                    existing_urls.add(result['url'])
            logger.info(f"Found {len(crawled_results)} articles from website crawl")
        
        # If we still need more results, try backup search
        if len(all_articles) < 10:
            logger.info("Supplementing with backup search...")
            backup_results = self._search_with_retry(topic, use_backup=True)
            if backup_results and backup_results.articles:
                for article in backup_results.articles:
                    if article.url not in existing_urls:
                        all_articles.append(article)
                        existing_urls.add(article.url)
                logger.info(f"Found {len(backup_results.articles)} articles from backup search")
        
        # Create final search results
        search_results = SearchResults(articles=all_articles)
        
        if len(search_results.articles) < 5:  # Reduced minimum requirement
            error_msg = f"Failed to gather sufficient sources. Only found {len(search_results.articles)} valid sources."
            logger.error(error_msg)
            yield RunResponse(
                event=RunEvent.run_completed,
                message=error_msg
            )
            return
        
        logger.info(f"Successfully gathered {len(search_results.articles)} unique sources for analysis")
        
        # Writing phase
        print("\nGenerating report from search results...")
        writer_response = self.writer.run(
            f"""Generate a comprehensive research report on: {topic}
            Use the following articles as sources:
            {json.dumps([{'title': a.title, 'url': a.url, 'description': a.description} for a in search_results.articles], indent=2)}
            
            Format the output in markdown with:
            1. Clear section headers using #, ##, ###
            2. Proper paragraph spacing
            3. Bullet points where appropriate
            4. Links to sources
            5. A references section at the end
            
            Focus on readability and proper markdown formatting.""",
            stream=False
        )
        
        if isinstance(writer_response, RunResponse):
            content = writer_response.content
        else:
            content = writer_response

        # Validate content
        if not self._validate_content(content):
            print("\nFirst attempt produced invalid content, trying again...")
            # Try one more time with a more structured prompt
            writer_response = self.writer.run(
                f"""Generate a clear, well-structured research report on: {topic}
                Format the output in proper markdown with:
                1. A main title using # 
                2. Section headers using ##
                3. Subsection headers using ###
                4. Well-formatted paragraphs
                5. Bullet points for lists
                6. A references section at the end
                
                Source articles:
                {json.dumps([{'title': a.title, 'url': a.url} for a in search_results.articles], indent=2)}""",
                stream=False
            )
            
            if isinstance(writer_response, RunResponse):
                content = writer_response.content
            else:
                content = writer_response
            
            if not self._validate_content(content):
                yield RunResponse(
                    event=RunEvent.run_completed,
                    message="Failed to generate readable content. Please try again."
                )
                return

        # Save as HTML
        html_file = self._save_markdown(topic, content)
        
        if not html_file:
            yield RunResponse(
                event=RunEvent.run_completed,
                message="Failed to save HTML file. Please try again."
            )
            return
        
        # Print the report to console and yield response
        print("\n=== Generated Report ===\n")
        print(content)
        print("\n=====================\n")
        
        yield RunResponse(
            event=RunEvent.run_completed,
            messages=f"Report generated successfully. HTML saved as: {html_file}",
            content=content
        )
        
        return

class WebsiteCrawler:
    """Crawler to extract relevant information from specified websites."""
    
    def __init__(self, max_pages_per_site: int = 10):
        self.max_pages_per_site = max_pages_per_site
        self.visited_urls: Set[str] = set()
        self.results: Dict[str, List[dict]] = {}
        self.file_handler = None
        
        # Set up logging
        self.relevance_log = []  # Store relevance decisions
    
    def _check_relevance(self, text: str, keywords: List[str]) -> tuple[bool, dict]:
        """
        Check if the page content is relevant based on keywords.
        Returns a tuple of (is_relevant, relevance_info).
        """
        text_lower = text.lower()
        keyword_matches = {}
        
        # Check each keyword and count occurrences
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = text_lower.count(keyword_lower)
            keyword_matches[keyword] = count
        
        # Page is relevant if any keyword is found
        is_relevant = any(count > 0 for count in keyword_matches.values())
        
        # Prepare relevance information
        relevance_info = {
            'is_relevant': is_relevant,
            'keyword_matches': keyword_matches,
            'total_matches': sum(keyword_matches.values()),
            'matching_keywords': [k for k, v in keyword_matches.items() if v > 0],
            'text_length': len(text)
        }
        
        return is_relevant, relevance_info

    def crawl_page(self, url: str, keywords: List[str]) -> List[dict]:
        """Crawl a single page and extract relevant information."""
        try:
            # Skip if already visited
            if url in self.visited_urls:
                logger.debug(f"Skipping already visited URL: {url}")
                return []
            
            self.visited_urls.add(url)
            logger.info(f"Crawling page: {url}")
            
            # Fetch and parse the page
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get page title
            title = soup.title.string if soup.title else url
            
            # Extract text content
            text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
            
            # Check relevance and get detailed information
            is_relevant, relevance_info = self._check_relevance(text, keywords)
            
            # Log relevance decision
            log_entry = {
                'url': url,
                'title': title,
                'timestamp': datetime.now().isoformat(),
                'relevance_info': relevance_info
            }
            self.relevance_log.append(log_entry)
            
            # Log the decision with details
            if is_relevant:
              logger.info(
                    f"Page is RELEVANT: {url}\n"
                    f"- Title: {title}\n"
                    f"- Matching keywords: {relevance_info['matching_keywords']}\n"
                    f"- Total matches: {relevance_info['total_matches']}"
                )
            else:
              logger.info(
                    f"Page is NOT RELEVANT: {url}\n"
                    f"- Title: {title}\n"
                    f"- Checked keywords: {keywords}\n"
                    f"- No keyword matches found in {relevance_info['text_length']} characters of text"
                )
            
            results = []
            if is_relevant:
                # Extract links for further crawling
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(url, href)
                    if self.is_valid_url(absolute_url):
                        links.append(absolute_url)
                
                # If page is relevant, process and download any supported files
                if self.file_handler:
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        absolute_url = urljoin(url, href)
                        if self.file_handler.is_supported_file(absolute_url):
                            downloaded_path = self.file_handler.download_file(absolute_url, source_page=url)
                            if downloaded_path:
                              logger.info(f"Downloaded file from relevant page: {absolute_url} to {downloaded_path}")
                
                # Store the relevant page information
                results.append({
                    'url': url,
                    'text': text,
                    'title': title,
                    'links': links,
                    'relevance_info': relevance_info
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            return []
    
    def save_relevance_log(self, output_dir: str):
        """Save the relevance log to a markdown file."""
        try:
            log_file = os.path.join(output_dir, 'crawl_relevance_log.md')
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("# Web Crawling Relevance Log\n\n")
                
                # Summary statistics
                total_pages = len(self.relevance_log)
                relevant_pages = sum(1 for entry in self.relevance_log if entry['relevance_info']['is_relevant'])
                
                f.write(f"## Summary\n")
                f.write(f"- Total pages crawled: {total_pages}\n")
                f.write(f"- Relevant pages found: {relevant_pages}\n")
                f.write(f"- Non-relevant pages: {total_pages - relevant_pages}\n\n")
                
                # Relevant pages
                f.write("## Relevant Pages\n\n")
                for entry in self.relevance_log:
                    if entry['relevance_info']['is_relevant']:
                        f.write(f"### {entry['title']}\n")
                        f.write(f"- URL: {entry['url']}\n")
                        f.write(f"- Matching keywords: {entry['relevance_info']['matching_keywords']}\n")
                        f.write(f"- Total matches: {entry['relevance_info']['total_matches']}\n")
                        f.write(f"- Crawled at: {entry['timestamp']}\n\n")
                
                # Non-relevant pages
                f.write("## Non-Relevant Pages\n\n")
                for entry in self.relevance_log:
                    if not entry['relevance_info']['is_relevant']:
                        f.write(f"### {entry['title']}\n")
                        f.write(f"- URL: {entry['url']}\n")
                        f.write(f"- Text length: {entry['relevance_info']['text_length']} characters\n")
                        f.write(f"- Crawled at: {entry['timestamp']}\n\n")
                
        except Exception as e:
          logger.error(f"Error saving relevance log: {str(e)}")

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to allowed domains."""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc and parsed.scheme in {'http', 'https'})
        except:
            return False
    
    def extract_text_and_links(self, url: str, soup: BeautifulSoup):
        """Extract relevant text and links from a page."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            links.append(absolute_url)
        return links
    
    def crawl_website(self, base_url: str, keywords: List[str]) -> List[dict]:
        """Crawl a website starting from the base URL."""
        to_visit = {base_url}
        results = []
        visited_count = 0
        
        while to_visit and visited_count < self.max_pages_per_site:
            url = to_visit.pop()
            page_results, links = self.crawl_page(url, keywords), self.extract_text_and_links(url, BeautifulSoup(requests.get(url, timeout=10).text, 'html.parser'))
            results.extend(page_results)
            
            # Add new links to visit
            domain = urlparse(base_url).netloc
            new_links = {link for link in links 
                        if urlparse(link).netloc == domain 
                        and link not in self.visited_urls}
            to_visit.update(new_links)
            visited_count += 1
        
        return results

    def crawl_all_websites(self, websites: List[str], keywords: List[str]) -> List[dict]:
        """Crawl multiple websites in parallel."""
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {
                executor.submit(self.crawl_website, url, keywords): url 
                for url in websites
            }
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"Completed crawling {url}, found {len(results)} relevant pages")
                except Exception as e:
                    logger.error(f"Failed to crawl {url}: {str(e)}")
        
        return all_results

# Create the workflow
searcher = Agent(
    model=Groq(
        id=SEARCHER_MODEL_CONFIG["id"],
        api_key=NVIDIA_API_KEY,
        temperature=SEARCHER_MODEL_CONFIG["temperature"],
        top_p=SEARCHER_MODEL_CONFIG["top_p"]
    ),
    tools=[DuckDuckGoTools(fixed_max_results=DUCK_DUCK_GO_FIXED_MAX_RESULTS)],
    instructions=[
        "Given a topic, search for 20 articles and return the 15 most relevant articles.",
        "For each article, provide:",
        "- title: The article title",
        "- url: The article URL",
        "- description: A brief description or summary",
        "Return the results in a structured format with these exact field names."
    ],
    # response_model=SearchResults,
    structured_outputs=True
)

backup_searcher = Agent(
    model=Groq(
        id=SEARCHER_MODEL_CONFIG["id"],
        api_key=NVIDIA_API_KEY,
        temperature=SEARCHER_MODEL_CONFIG["temperature"],
        top_p=SEARCHER_MODEL_CONFIG["top_p"]
    ),
    tools=[GoogleSearchTools()],
    instructions=[
        "Given a topic, search for 20 articles and return the 15 most relevant articles.",
        "For each article, provide:",
        "- title: The article title",
        "- url: The article URL",
        "- description: A brief description or summary",
        "Return the results in a structured format with these exact field names."
    ],
    # response_model=SearchResults,
    structured_outputs=True
)

writer = Agent(
    model=Groq(
        id=WRITER_MODEL_CONFIG["id"],
        api_key=NVIDIA_API_KEY,
        temperature=WRITER_MODEL_CONFIG["temperature"],
        top_p=WRITER_MODEL_CONFIG["top_p"],
    ),
    instructions=[
        "You are a professional research analyst tasked with creating a comprehensive report on the given topic.",
        "The sources provided include both general web search results and specialized intelligence/security websites.",
        "Carefully analyze and cross-reference information from all sources to create a detailed report.",
        "",
        "Report Structure:",
        "1. Executive Summary (2-3 paragraphs)",
        "   - Provide a clear, concise overview of the main findings",
        "   - Address the research question directly",
        "   - Highlight key discoveries and implications",
        "",
        "2. Detailed Analysis (Multiple sections)",
        "   - Break down the topic into relevant themes or aspects",
        "   - For each theme:",
        "     * Present detailed findings from multiple sources",
        "     * Cross-reference information between general and specialized sources",
        "     * Analyze trends, patterns, and developments",
        "     * Discuss implications and potential impacts",
        "",
        "3. Source Analysis and Credibility",
        "   For each major source:",
        "   - Evaluate source credibility and expertise",
        "   - Note if from specialized intelligence/security website",
        "   - Assess potential biases or limitations",
        "   - Key findings and unique contributions",
        "",
        "4. Key Takeaways and Strategic Implications",
        "   - Synthesize findings from all sources",
        "   - Compare/contrast general media vs specialized analysis",
        "   - Discuss broader geopolitical implications",
        "   - Address potential future developments",
        "",
        "5. References",
        "   - Group sources by type (specialized websites vs general media)",
        "   - List all sources with full citations",
        "   - Include URLs as clickable markdown links [Title](URL)",
        "   - Ensure every major claim has at least one linked source",
        "",
        "Important Guidelines:",
        "- Prioritize information from specialized intelligence/security sources",
        "- Cross-validate claims between multiple sources when possible",
        "- Maintain a professional, analytical tone",
        "- Support all claims with evidence",
        "- Include specific examples and data points",
        "- Use direct quotes for significant statements",
        "- Address potential biases in reporting",
        "- Ensure the report directly answers the research question",
        "",
        "Format the report with clear markdown headings (# ## ###), subheadings, and paragraphs.",
        "Each major section should contain multiple paragraphs with detailed analysis."
    ],
    structured_outputs=True
)

generate_blog_post = BlogPostGenerator(
    session_id=f"generate-blog-post-on-{topic}",
    searcher=searcher,
    backup_searcher=backup_searcher,
    writer=writer,
    file_handler=None,
    initial_websites=INITIAL_WEBSITES.copy(),  # Added this line
    storage=SqliteWorkflowStorage(
        table_name="generate_blog_post_workflows",
        db_file="tmp/workflows.db",
    ),
)

# Run workflow
blog_post: Iterator[RunResponse] = generate_blog_post.run(topic=topic, use_cache=False)

# Print the response
pprint_run_response(blog_post, markdown=True)