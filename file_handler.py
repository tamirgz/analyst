import os
import requests
from typing import Optional, List, Set
from urllib.parse import urlparse, unquote
from pathlib import Path
from datetime import datetime
from save_report import save_markdown_report
from agno.utils.log import logger


class FileHandler:
    """Handler for downloading and saving files discovered during web crawling."""
    
    SUPPORTED_EXTENSIONS = {
        'pdf': 'application/pdf',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'csv': 'text/csv'
    }
    
    # Common browser headers
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    def __init__(self):
        # Get the report directory for the current date
        self.report_dir, _ = save_markdown_report()
        self.downloaded_files: Set[str] = set()
        self.file_metadata: List[dict] = []
        self.failed_downloads: List[dict] = []  # Track failed downloads
        
        # Create a subdirectory for downloaded files
        self.downloads_dir = os.path.join(self.report_dir, 'downloads')
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        # Create a metadata file to track downloaded files
        self.metadata_file = os.path.join(self.downloads_dir, 'files_metadata.md')

    def is_supported_file(self, url: str) -> bool:
        """Check if the URL points to a supported file type."""
        parsed_url = urlparse(url)
        extension = os.path.splitext(parsed_url.path)[1].lower().lstrip('.')
        return extension in self.SUPPORTED_EXTENSIONS
    
    def get_filename_from_url(self, url: str, content_type: Optional[str] = None) -> str:
        """Generate a safe filename from the URL."""
        # Get the filename from the URL
        parsed_url = urlparse(url)
        filename = os.path.basename(unquote(parsed_url.path))
        
        # If no filename in URL, create one based on content type
        if not filename:
            extension = next(
                (ext for ext, mime in self.SUPPORTED_EXTENSIONS.items() 
                 if mime == content_type),
                'unknown'
            )
            filename = f"downloaded_file.{extension}"
        
        # Ensure filename is safe and unique
        safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-')
        base, ext = os.path.splitext(safe_filename)
        
        # Add number suffix if file exists
        counter = 1
        while os.path.exists(os.path.join(self.downloads_dir, safe_filename)):
            safe_filename = f"{base}_{counter}{ext}"
            counter += 1
        
        return safe_filename
    
    def download_file(self, url: str, source_page: str = None) -> Optional[str]:
        """
        Download a file from the URL and save it to the downloads directory.
        Returns the path to the saved file if successful, None otherwise.
        """
        if url in self.downloaded_files:
            logger.info(f"File already downloaded: {url}")
            return None
        
        try:
            # Create a session to maintain headers across redirects
            session = requests.Session()
            session.headers.update(self.HEADERS)
            
            # First make a HEAD request to check content type and size
            head_response = session.head(url, timeout=10, allow_redirects=True)
            head_response.raise_for_status()
            
            content_type = head_response.headers.get('content-type', '').lower().split(';')[0]
            content_length = int(head_response.headers.get('content-length', 0))
            
            # Check if content type is supported and size is reasonable (less than 100MB)
            if not any(mime in content_type for mime in self.SUPPORTED_EXTENSIONS.values()):
                logger.warning(f"Unsupported content type: {content_type} for URL: {url}")
                return None
            
            if content_length > 100 * 1024 * 1024:  # 100MB limit
                logger.warning(f"File too large ({content_length} bytes) for URL: {url}")
                return None
            
            # Make the actual download request
            response = session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Generate safe filename
            filename = self.get_filename_from_url(url, content_type)
            file_path = os.path.join(self.downloads_dir, filename)
            
            # Save the file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Record metadata
            metadata = {
                'filename': filename,
                'source_url': url,
                'source_page': source_page,
                'content_type': content_type,
                'download_time': datetime.now().isoformat(),
                'file_size': os.path.getsize(file_path)
            }
            self.file_metadata.append(metadata)
            
            # Update metadata file
            self._update_metadata_file()
            
            self.downloaded_files.add(url)
            logger.info(f"Successfully downloaded: {url} to {file_path}")
            return file_path
            
        except requests.RequestException as e:
            error_info = {
                'url': url,
                'source_page': source_page,
                'error': str(e),
                'time': datetime.now().isoformat()
            }
            self.failed_downloads.append(error_info)
            self._update_metadata_file()  # Update metadata including failed downloads
            logger.error(f"Error downloading file from {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while downloading {url}: {str(e)}")
            return None
    
    def _update_metadata_file(self):
        """Update the metadata markdown file with information about downloaded files."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                f.write("# Downloaded Files Metadata\n\n")
                
                # Successful downloads
                if self.file_metadata:
                    f.write("## Successfully Downloaded Files\n\n")
                    for metadata in self.file_metadata:
                        f.write(f"### {metadata['filename']}\n")
                        f.write(f"- Source URL: {metadata['source_url']}\n")
                        if metadata['source_page']:
                            f.write(f"- Found on page: {metadata['source_page']}\n")
                        f.write(f"- Content Type: {metadata['content_type']}\n")
                        f.write(f"- Download Time: {metadata['download_time']}\n")
                        f.write(f"- File Size: {metadata['file_size']} bytes\n\n")
                
                # Failed downloads
                if self.failed_downloads:
                    f.write("## Failed Downloads\n\n")
                    for failed in self.failed_downloads:
                        f.write(f"### {failed['url']}\n")
                        if failed['source_page']:
                            f.write(f"- Found on page: {failed['source_page']}\n")
                        f.write(f"- Error: {failed['error']}\n")
                        f.write(f"- Time: {failed['time']}\n\n")
                
        except Exception as e:
            logger.error(f"Error updating metadata file: {str(e)}")

    def get_downloaded_files(self) -> List[str]:
        """Return a list of all downloaded file paths."""
        return [os.path.join(self.downloads_dir, f) 
                for f in os.listdir(self.downloads_dir)
                if os.path.isfile(os.path.join(self.downloads_dir, f))]
