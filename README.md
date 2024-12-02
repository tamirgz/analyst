# Web Research and Report Generation System

An advanced AI-powered system for automated web research and report generation. This system uses AI agents to search, analyze, and compile comprehensive reports on any given topic.

## Features

- **Intelligent Web Search**
  - Multi-source search using DuckDuckGo and Google
  - Smart retry mechanism with rate limit handling
  - Configurable search depth and result limits
  - Domain filtering for trusted sources

- **Advanced Report Generation**
  - Beautiful HTML reports with modern styling
  - Automatic keyword extraction
  - Source validation and relevance scoring
  - Comprehensive logging of research process

- **Smart Caching**
  - Caches search results for faster repeat queries
  - Configurable cache directory
  - Cache invalidation management

- **Error Handling**
  - Graceful fallback between search engines
  - Rate limit detection and backoff
  - Detailed error logging
  - Automatic retry mechanisms

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd phidata_analyst
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
   - Create a `.env` file in the project root
   - Add your API keys:
     ```
     NVIDIA_API_KEY=your-nvidia-api-key
     GOOGLE_API_KEY=your-google-api-key
     ```

## Usage

### Basic Usage

```python
from web_search import create_blog_post_workflow

# Create a workflow instance
workflow = create_blog_post_workflow()

# Generate a report
for response in workflow.run("Your research topic"):
    print(response.message)
```

### Advanced Usage

```python
from web_search import BlogPostGenerator, SqlWorkflowStorage
from phi.llm import Nvidia
from phi.tools import DuckDuckGo, GoogleSearch

# Configure custom agents
searcher = Agent(
    model=Nvidia(
        id="meta/llama-3.2-3b-instruct",
        temperature=0.3,
        top_p=0.1
    ),
    tools=[DuckDuckGo(fixed_max_results=10)]
)

# Initialize with custom configuration
generator = BlogPostGenerator(
    searcher=searcher,
    storage=SqlWorkflowStorage(
        table_name="custom_workflows",
        db_file="path/to/db.sqlite"
    )
)

# Run with caching enabled
for response in generator.run("topic", use_cache=True):
    print(response.message)
```

## Output

The system generates:
1. Professional HTML reports with:
   - Executive summary
   - Detailed analysis
   - Source citations
   - Generation timestamp
2. Detailed logs of:
   - Search process
   - Keyword extraction
   - Source relevance
   - Download attempts

Reports are saved in:
- Default: `./reports/YYYY-MM-DD-HH-MM-SS/`
- Custom: Configurable via `file_handler`

## Configuration

Key configuration options:

```python
DUCK_DUCK_GO_FIXED_MAX_RESULTS = 10  # Max results from DuckDuckGo
DEFAULT_TEMPERATURE = 0.3             # Model temperature
TOP_P = 0.1                          # Top-p sampling parameter
```

Trusted domains can be configured in `BlogPostGenerator.trusted_domains`.

## Logging

The system uses `phi.utils.log` for comprehensive logging:
- Search progress and results
- Keyword extraction details
- File downloads and failures
- Report generation status

Logs are color-coded for easy monitoring:
- INFO: Normal operations
- WARNING: Non-critical issues
- ERROR: Critical failures

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Phi](https://github.com/phidatahq/phidata)
- Uses NVIDIA AI models
- Search powered by DuckDuckGo and Google
