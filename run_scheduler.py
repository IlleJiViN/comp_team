import schedule
import time
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_scraper():
    logging.info("Starting async_scraper.py...")
    try:
        # Run the scraper
        result = subprocess.run([".venv/Scripts/python", "async_scraper.py"], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("Scraping completed successfully.")
        else:
            logging.error(f"Scraping failed with return code {result.returncode}.")
            logging.error(f"Error Output: {result.stderr}")
    except Exception as e:
        logging.error(f"Failed to run scraper: {e}")

# Schedule to run every day at 3:00 AM
schedule.every().day.at("03:00").do(run_scraper)

logging.info("Scheduler started. async_scraper.py is scheduled to run daily at 03:00 AM.")

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60) # check every minute
