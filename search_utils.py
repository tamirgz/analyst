import time
import logging
import random
import threading
from typing import Optional, Dict, Any
from duckduckgo_search.exceptions import RatelimitException

logger = logging.getLogger(__name__)

class RateLimitedSearch:
    """Rate limited search implementation with exponential backoff."""
    
    def __init__(self):
        self.last_request_time = 0
        self.min_delay = 30  # Increased minimum delay between requests to 30 seconds
        self.max_delay = 300  # Maximum delay of 5 minutes
        self.jitter = 5  # Added more jitter range
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5  # Increased max failures before giving up
        self._delay_lock = threading.Lock()  # Add thread safety
        
    def _add_jitter(self, delay: float) -> float:
        """Add randomized jitter to delay."""
        return delay + random.uniform(-self.jitter, self.jitter)
    
    def _wait_for_rate_limit(self):
        """Wait for rate limit with exponential backoff."""
        with self._delay_lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            
            # Calculate delay based on consecutive failures
            if self.consecutive_failures > 0:
                delay = min(
                    self.max_delay,
                    self.min_delay * (2 ** (self.consecutive_failures - 1))
                )
            else:
                delay = self.min_delay
                
            # Add jitter to prevent synchronized requests
            jitter = random.uniform(-self.jitter, self.jitter)
            delay = max(0, delay + jitter)
            
            # If not enough time has elapsed, wait
            if elapsed < delay:
                time.sleep(delay - elapsed)
                
            self.last_request_time = time.time()
            
    def execute_with_retry(self, 
                          search_func: callable, 
                          max_retries: int = 3,
                          **kwargs) -> Optional[Dict[str, Any]]:
        """Execute search with retries and exponential backoff."""
        
        for attempt in range(max_retries):
            try:
                # Enforce rate limiting
                self._wait_for_rate_limit()
                
                # Execute search
                result = search_func(**kwargs)
                
                # Reset consecutive failures on success
                self.consecutive_failures = 0
                return result
                
            except RatelimitException as e:
                self.consecutive_failures += 1
                
                # Calculate backoff time
                backoff = min(
                    self.max_delay,
                    self.min_delay * (2 ** attempt)
                )
                backoff = self._add_jitter(backoff)
                
                if attempt == max_retries - 1:
                    logger.error(f"Rate limit exceeded after {max_retries} retries")
                    raise
                    
                logger.warning(f"Rate limit hit, attempt {attempt + 1}/{max_retries}. "
                             f"Waiting {backoff:.2f} seconds...")
                time.sleep(backoff)
                
                # If we've hit too many consecutive failures, raise an exception
                if self.consecutive_failures >= self.max_consecutive_failures:
                    logger.error("Too many consecutive rate limit failures")
                    raise RatelimitException("Persistent rate limiting detected")
                continue
                
            except Exception as e:
                logger.error(f"Search error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                    
                backoff = self.min_delay * (2 ** attempt)
                backoff = self._add_jitter(backoff)
                logger.info(f"Retrying in {backoff:.2f} seconds...")
                time.sleep(backoff)
                
        return None
