
import asyncio
import sys
import os

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.pdf_downloader import download_pdf

async def test_download():
    # URL that failed in logs (TechRxiv)
    url = "https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.170956672.21573677"
    print(f"Attempting to download: {url}")
    
    try:
        result = await download_pdf(url)
        
        if result.success:
            print(f"SUCCESS! Downloaded {result.filesize} bytes.")
            print(f"Filename: {result.filename}")
        else:
            print(f"FAILED: {result.error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        sys.exit(1)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test_download())
    loop.close()
