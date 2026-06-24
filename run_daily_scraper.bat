@echo off
echo Starting Spotsync Daily Async Scraper...
cd /d "C:\Users\dev\gravity"
call .venv\Scripts\activate
python async_scraper.py
echo Scraper finished.
exit
