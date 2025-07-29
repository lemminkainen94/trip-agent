"""Main script to run the Trip Agent system."""

import os
import argparse
import subprocess
import sys
from dotenv import load_dotenv


def run_api():
    """Run the FastAPI application."""
    from src.api.app import app
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=port)


def run_ui():
    """Run the Streamlit UI."""
    subprocess.run(["streamlit", "run", "src/ui/app.py"])


def run_both():
    """Run both the API and UI in separate processes."""
    import multiprocessing
    
    # Create processes
    api_process = multiprocessing.Process(target=run_api)
    ui_process = multiprocessing.Process(target=run_ui)
    
    # Start processes
    api_process.start()
    ui_process.start()
    
    try:
        # Wait for processes to finish
        api_process.join()
        ui_process.join()
    except KeyboardInterrupt:
        # Handle keyboard interrupt
        print("Shutting down...")
        api_process.terminate()
        ui_process.terminate()
        api_process.join()
        ui_process.join()


def main():
    """Main function to run the Trip Agent system."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trip Agent - Multi-Agent Trip Planner")
    parser.add_argument(
        "--mode",
        choices=["api", "ui", "both"],
        default="both",
        help="Run mode: api, ui, or both (default: both)"
    )
    
    args = parser.parse_args()
    
    # Run in the specified mode
    if args.mode == "api":
        run_api()
    elif args.mode == "ui":
        run_ui()
    else:
        run_both()


if __name__ == "__main__":
    main()
    # you can use this prompt for test:
        # I like museums and galleries and parks. I'm a foodie who loves food markets and food halls. I like coffee. 
        # I'd like to go to a club at least once during the trip. 
        # I like moderate intensity touring and have mid-range budget. I'm travelling alone