from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Add root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
RAGPY_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(RAGPY_ROOT))

from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    # It returns HTML, so we check content type
    assert "text/html" in response.headers["content-type"]

def test_get_credentials():
    response = client.get("/get_credentials")
    assert response.status_code == 200
    data = response.json()
    assert "OPENAI_API_KEY" in data
    assert "PINECONE_API_KEY" in data

def test_upload_csv_invalid_extension():
    # Test uploading a non-csv file
    files = {'file': ('test.txt', b'some content', 'text/plain')}
    response = client.post("/upload_csv", files=files)
    assert response.status_code == 400
    assert response.json() == {"error": "Only .csv files are accepted."}

def test_upload_csv_success():
    # Test uploading a valid csv file
    csv_content = b"header1,header2\nval1,val2"
    files = {'file': ('test.csv', csv_content, 'text/csv')}
    response = client.post("/upload_csv", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "path" in data
    assert "tree" in data
    assert "output.csv" in data["tree"]
