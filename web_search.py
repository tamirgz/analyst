import json
import re
import time
import logging
import os
import concurrent.futures
from typing import Optional, Iterator, List, Set, Dict
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

# Phi imports
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.agent import Agent
from phi.model.nvidia import Nvidia  
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
from phi.utils.pprint import pprint_run_response
from phi.utils.log import logger

# Error handling imports
from duckduckgo_search.exceptions import RatelimitException
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from requests.exceptions import HTTPError

# The topic to generate a blog post on
topic = "Is there a process of establishment of Israeli Military or Offensive Cyber Industry in Australia?"

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
    initial_websites: List[str] = Field(default_factory=lambda: [
        "https://www.bellingcat.com/",
        "https://worldview.stratfor.com/",
        "https://thesoufancenter.org/",
        "https://www.globalsecurity.org/",
        "https://www.defenseone.com/"
    ])

    def __init__(
        self,
        session_id: str,
        searcher: Agent,
        backup_searcher: Agent,
        writer: Agent,
        storage: Optional[SqlWorkflowStorage] = None,
    ):
        super().__init__(
            session_id=session_id,
            searcher=searcher,
            backup_searcher=backup_searcher,
            writer=writer,
            storage=storage,
        )
        
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
            model=Nvidia(
                id="meta/llama-3.2-3b-instruct",
                api_key="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"
            ),
            tools=[DuckDuckGo(fixed_max_results=5)],
            instructions=search_instructions,
            response_model=SearchResults
        )
        
        # Backup searcher using Google Search
        self.backup_searcher = Agent(
            model=Nvidia(
                id="meta/llama-3.2-3b-instruct",
                api_key="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"
            ),
            tools=[GoogleSearch()],
            instructions=search_instructions,
            response_model=SearchResults
        )

        # Writer agent configuration
        writer_instructions = [
            "You are a professional research analyst tasked with creating a comprehensive report on the given topic.",
            "Carefully analyze each provided article and synthesize the information into a detailed report.",
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
            "     * Analyze trends, patterns, and developments",
            "     * Discuss implications and potential impacts",
            "",
            "3. Source Analysis",
            "   For each major article:",
            "   - Title and source credibility assessment",
            "   - Key findings and contributions to the topic",
            "   - Critical evaluation of the information",
            "",
            "4. Key Takeaways and Implications (2-3 paragraphs)",
            "   - Synthesize the most important findings",
            "   - Discuss broader implications",
            "   - Address potential future developments",
            "",
            "5. References",
            "   - List all sources with full citations",
            "   - Include URLs for all referenced articles",
            "   - Format URLs as clickable markdown links [Title](URL)",
            "   - Ensure every major claim has at least one linked source",
            "   - Include any additional relevant URLs discovered",
            "",
            "Important Guidelines:",
            "- Always maintain a professional, analytical tone",
            "- Support all claims with evidence from the sources",
            "- Critically evaluate the reliability of sources",
            "- Provide specific examples and data points",
            "- Include direct quotes when particularly relevant",
            "- Address potential biases or limitations in the research",
            "- Ensure the report directly answers the research question",
            "",
            "Format the report with clear headings, subheadings, and paragraphs for readability.",
            "Each major section should contain multiple paragraphs with detailed analysis."
        ]
        
        self.writer = Agent(
            model=Nvidia(
                id="meta/llama-3.2-3b-instruct",
                api_key="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"
            ),
            instructions=writer_instructions,
            structured_outputs=True
        )

    def _parse_search_response(self, response) -> Optional[SearchResults]:
        """Parse and validate search response into SearchResults model."""
        try:
            if isinstance(response, str):
                # Clean up markdown code blocks and extract JSON
                content = response.strip()
                if '```' in content:
                    # Extract content between code block markers
                    match = re.search(r'```(?:json)?\n(.*?)\n```', content, re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                    else:
                        # If no proper code block found, remove all ``` markers
                        content = re.sub(r'```(?:json)?\n?', '', content)
                        content = content.strip()
                
                # Try to parse JSON response
                try:
                    # Clean up any trailing commas before closing brackets/braces
                    content = re.sub(r',(\s*[}\]])', r'\1', content)
                    # Fix invalid escape sequences
                    content = re.sub(r'\\([^"\\\/bfnrtu])', r'\1', content)  # Remove invalid escapes
                    content = content.replace('\t', ' ')  # Replace tabs with spaces
                    # Handle any remaining unicode escapes
                    content = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), content)
                    
                    data = json.loads(content)
                    
                    if isinstance(data, dict) and 'articles' in data:
                        articles = []
                        for article in data['articles']:
                            if isinstance(article, dict):
                                # Ensure all required fields are strings
                                article = {
                                    'title': str(article.get('title', '')).strip(),
                                    'url': str(article.get('url', '')).strip(),
                                    'description': str(article.get('description', '')).strip()
                                }
                                if article['title'] and article['url']:  # Only add if has required fields
                                    articles.append(NewsArticle(**article))
                        
                        if articles:
                            logger.info(f"Successfully parsed {len(articles)} articles from JSON")
                            return SearchResults(articles=articles)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {str(e)}, attempting to extract data manually")
                    
                # Fallback to regex extraction if JSON parsing fails
                urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', content)
                titles = re.findall(r'"title":\s*"([^"]+)"', content)
                descriptions = re.findall(r'"description":\s*"([^"]+)"', content)
                
                if not urls:  # Try alternative patterns
                    urls = re.findall(r'(?<=\()http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+(?=\))', content)
                
                if urls:
                    articles = []
                    for i, url in enumerate(urls):
                        title = titles[i] if i < len(titles) else f"Article {i+1}"
                        description = descriptions[i] if i < len(descriptions) else ""
                        # Clean up extracted data
                        title = title.strip().replace('\\"', '"')
                        url = url.strip().replace('\\"', '"')
                        description = description.strip().replace('\\"', '"')
                        
                        if url:  # Only add if URL exists
                            articles.append(NewsArticle(
                                title=title,
                                url=url,
                                description=description
                            ))
                    
                    if articles:
                        logger.info(f"Successfully extracted {len(articles)} articles using regex")
                        return SearchResults(articles=articles)
                    
                logger.warning("No valid articles found in response")
                return None
                
            elif isinstance(response, dict):
                # Handle dictionary response
                if 'articles' in response:
                    articles = []
                    for article in response['articles']:
                        if isinstance(article, dict):
                            # Ensure all fields are strings
                            article = {
                                'title': str(article.get('title', '')).strip(),
                                'url': str(article.get('url', '')).strip(),
                                'description': str(article.get('description', '')).strip()
                            }
                            if article['title'] and article['url']:
                                articles.append(NewsArticle(**article))
                        elif isinstance(article, NewsArticle):
                            articles.append(article)
                    
                    if articles:
                        logger.info(f"Successfully processed {len(articles)} articles from dict")
                        return SearchResults(articles=articles)
                return None
                
            elif isinstance(response, SearchResults):
                # Already in correct format
                return response
                
            elif isinstance(response, RunResponse):
                # Extract from RunResponse
                if response.content:
                    return self._parse_search_response(response.content)
                return None
                
            logger.error(f"Unsupported response type: {type(response)}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing search response: {str(e)}")
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
                        # Try backup search if primary is rate limited
                        backup_results = self._search_with_retry(topic, use_backup=True, max_retries=max_retries)
                        if backup_results:
                            return backup_results
                    # If both sources are rate limited, use longer backoff
                    backoff_time = min(3600, 60 * (2 ** attempt))  # Max 1 hour backoff
                    logger.info(f"All search methods rate limited. Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
                
                logger.info(f"\nAttempting {source} search (attempt {attempt + 1}/{max_retries})...")
                
                # Try different search prompts to improve results
                search_prompts = [
                    f"""Search for detailed articles about: {topic}
                    Return only high-quality, relevant sources.
                    Format the results as a JSON object with an 'articles' array containing:
                    - title: The article title
                    - url: The article URL
                    - description: A brief description or summary
                    """,
                    f"""Find comprehensive articles and research papers about: {topic}
                    Focus on authoritative sources and recent publications.
                    Return results in JSON format with 'articles' array.
                    """,
                    f"""Locate detailed analysis and reports discussing: {topic}
                    Prioritize academic, industry, and news sources.
                    Return structured JSON with article details.
                    """
                ]
                
                # Try each prompt until we get results
                for prompt in search_prompts:
                    try:
                        response = searcher.run(prompt, stream=False)
                        results = self._parse_search_response(response)
                        if results and results.articles:
                            logger.info(f"Found {len(results.articles)} articles from {source} search")
                            return results
                    except Exception as e:
                        if any(err in str(e).lower() for err in ["rate", "limit", "quota", "exhausted"]):
                            rate_limited_sources.add(source)
                            raise
                        logger.warning(f"Search prompt failed: {str(e)}")
                        continue
                
                logger.warning(f"{source.title()} search returned no valid results")
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(err in error_msg for err in ["rate", "limit", "quota", "exhausted"]):
                    rate_limited_sources.add(source)
                    logger.error(f"{source} search rate limited: {str(e)}")
                    # Try alternative source immediately
                    if not use_backup:
                        backup_results = self._search_with_retry(topic, use_backup=True, max_retries=max_retries)
                        if backup_results:
                            return backup_results
                else:
                    logger.error(f"Error during {source} search (attempt {attempt + 1}): {str(e)}")
                
                if attempt < max_retries - 1:
                    backoff_time = 2 ** attempt
                    if source in rate_limited_sources:
                        backoff_time = min(3600, 60 * (2 ** attempt))  # Longer backoff for rate limits
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
        """Save the content as a markdown file."""
        # Create filename from topic
        filename = re.sub(r'[^\w\s-]', '', topic.lower())  # Remove special chars
        filename = re.sub(r'[-\s]+', '-', filename)        # Replace spaces with hyphens
        filename = f"{filename}.md"
        
        # Ensure the content has proper markdown formatting
        formatted_content = content
        
        # If content doesn't start with a title, add it
        if not content.startswith('# '):
            formatted_content = f"# {topic}\n\n{content}"
        
        # Write the markdown file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            logger.info(f"Successfully saved markdown file: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save markdown file: {str(e)}")
            return None

    def run(self, topic: str, use_cache: bool = True) -> Iterator[RunResponse]:
        """Run the blog post generation workflow."""
        logger.info(f"Starting blog post generation for topic: {topic}")
        
        # Extract keywords from topic
        keywords = topic.lower().split()
        keywords = [w for w in keywords if len(w) > 3 and w not in {'what', 'where', 'when', 'how', 'why', 'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should', 'the', 'and', 'but', 'or', 'for', 'with'}]
        
        # First, crawl initial websites
        crawler = WebsiteCrawler(max_pages_per_site=10)
        crawled_results = crawler.crawl_all_websites(self.initial_websites, keywords)
        
        if crawled_results:
            logger.info(f"Found {len(crawled_results)} relevant pages from initial websites")
            search_results = SearchResults(articles=[NewsArticle(**result) for result in crawled_results])
        else:
            logger.info("No results from initial websites, falling back to search")
            search_results = self._search_with_retry(topic)
        
        # If initial crawl yields insufficient results, supplement with search
        if not search_results or len(search_results.articles) < 10:
            logger.info("Supplementing results with additional search")
            additional_results = self._search_with_retry(topic)
            if additional_results:
                existing_urls = {article.url for article in search_results.articles} if search_results else set()
                for article in additional_results.articles:
                    if article.url not in existing_urls:
                        if not search_results:
                            search_results = SearchResults(articles=[])
                        search_results.articles.append(article)
                        existing_urls.add(article.url)
        
        if not search_results or len(search_results.articles) < 10:
            error_msg = f"Failed to gather sufficient sources. Only found {len(search_results.articles) if search_results else 0} valid sources."
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

        # Save as markdown
        markdown_file = self._save_markdown(topic, content)
        
        if not markdown_file:
            yield RunResponse(
                event=RunEvent.run_completed,
                message="Failed to save markdown file. Please try again."
            )
            return
        
        # Print the report to console and yield response
        print("\n=== Generated Report ===\n")
        print(content)
        print("\n=====================\n")
        
        yield RunResponse(
            event=RunEvent.run_completed,
            message=f"Report generated successfully. Markdown saved as: {markdown_file}",
            content=content
        )
        
        return

class WebsiteCrawler:
    """Crawler to extract relevant information from specified websites."""
    
    def __init__(self, max_pages_per_site: int = 10):
        self.max_pages_per_site = max_pages_per_site
        self.visited_urls: Set[str] = set()
        self.results: Dict[str, List[dict]] = {}
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to allowed domains."""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc and parsed.scheme in {'http', 'https'})
        except:
            return False
    
    def extract_text_and_links(self, url: str, soup: BeautifulSoup) -> tuple:
        """Extract relevant text and links from a page."""
        # Extract main content (adjust selectors based on site structure)
        content_selectors = ['article', 'main', '.content', '#content', '.post']
        text = ''
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                text = content.get_text(separator=' ', strip=True)
                break
        if not text:
            text = soup.body.get_text(separator=' ', strip=True)
        
        # Extract links
        links = []
        for a in soup.find_all('a', href=True):
            href = urljoin(url, a['href'])
            if self.is_valid_url(href):
                links.append(href)
        
        return text, links
    
    def crawl_page(self, url: str, keywords: List[str]) -> List[dict]:
        """Crawl a single page and extract relevant information."""
        if url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        results = []
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else ''
            
            # Extract text and links
            text, links = self.extract_text_and_links(url, soup)
            
            # Check relevance
            text_lower = text.lower()
            if any(keyword.lower() in text_lower for keyword in keywords):
                results.append({
                    'title': title,
                    'url': url,
                    'description': text[:500] + '...' if len(text) > 500 else text
                })
            
            # Return both results and links for further crawling
            return results, links
            
        except Exception as e:
            logger.warning(f"Failed to crawl {url}: {str(e)}")
            return [], []
    
    def crawl_website(self, base_url: str, keywords: List[str]) -> List[dict]:
        """Crawl a website starting from the base URL."""
        to_visit = {base_url}
        results = []
        visited_count = 0
        
        while to_visit and visited_count < self.max_pages_per_site:
            url = to_visit.pop()
            page_results, links = self.crawl_page(url, keywords)
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
    model=Nvidia(
        id="meta/llama-3.2-3b-instruct",
        api_key="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"
    ),
    tools=[DuckDuckGo(fixed_max_results=5)],
    instructions=[
        "Given a topic, search for 20 articles and return the 15 most relevant articles.",
        "For each article, provide:",
        "- title: The article title",
        "- url: The article URL",
        "- description: A brief description or summary",
        "Return the results in a structured format with these exact field names."
    ],
    response_model=SearchResults
)

backup_searcher = Agent(
    model=Nvidia(
        id="meta/llama-3.2-3b-instruct",
        api_key="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"
    ),
    tools=[GoogleSearch()],
    instructions=[
        "Given a topic, search for 20 articles and return the 15 most relevant articles.",
        "For each article, provide:",
        "- title: The article title",
        "- url: The article URL",
        "- description: A brief description or summary",
        "Return the results in a structured format with these exact field names."
    ],
    response_model=SearchResults
)

writer = Agent(
    model=Nvidia(
        id="meta/llama-3.2-3b-instruct",
        api_key="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"
    ),
    instructions=[
        "You are a professional research analyst tasked with creating a comprehensive report on the given topic.",
        "Carefully analyze each provided article and synthesize the information into a detailed report.",
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
        "     * Analyze trends, patterns, and developments",
        "     * Discuss implications and potential impacts",
        "",
        "3. Source Analysis",
        "   For each major article:",
        "   - Title and source credibility assessment",
        "   - Key findings and contributions to the topic",
        "   - Critical evaluation of the information",
        "",
        "4. Key Takeaways and Implications (2-3 paragraphs)",
        "   - Synthesize the most important findings",
        "   - Discuss broader implications",
        "   - Address potential future developments",
        "",
        "5. References",
        "   - List all sources with full citations",
        "   - Include URLs for all referenced articles",
        "   - Format URLs as clickable markdown links [Title](URL)",
        "   - Ensure every major claim has at least one linked source",
        "   - Include any additional relevant URLs discovered",
        "",
        "Important Guidelines:",
        "- Always maintain a professional, analytical tone",
        "- Support all claims with evidence from the sources",
        "- Critically evaluate the reliability of sources",
        "- Provide specific examples and data points",
        "- Include direct quotes when particularly relevant",
        "- Address potential biases or limitations in the research",
        "- Ensure the report directly answers the research question",
        "",
        "Format the report with clear headings, subheadings, and paragraphs for readability.",
        "Each major section should contain multiple paragraphs with detailed analysis."
    ],
    structured_outputs=True
)

generate_blog_post = BlogPostGenerator(
    session_id=f"generate-blog-post-on-{topic}",
    searcher=searcher,
    backup_searcher=backup_searcher,
    writer=writer,
    storage=SqlWorkflowStorage(
        table_name="generate_blog_post_workflows",
        db_file="tmp/workflows.db",
    ),
)

# Run workflow
blog_post: Iterator[RunResponse] = generate_blog_post.run(topic=topic, use_cache=False)

# Print the response
pprint_run_response(blog_post, markdown=True)
