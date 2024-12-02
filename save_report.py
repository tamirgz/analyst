import os
from datetime import datetime
import shutil

def save_markdown_report():
    # Get current date in YYYY-MM-DD format
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create directory name
    report_dir = f"report_{current_date}"
    
    # Create full path
    base_path = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(base_path, report_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(report_path, exist_ok=True)
    
    # Create markdown file path
    report_file = os.path.join(report_path, f"report_{current_date}.md")
    
    return report_path, report_file

if __name__ == "__main__":
    report_path, report_file = save_markdown_report()
    print(f"Report directory created at: {report_path}")
    print(f"Report file path: {report_file}")
