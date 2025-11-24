#!/usr/bin/env python3
"""
Quick test script for SSE endpoints.
"""
import requests
import sys

def test_sse(endpoint, data):
    """Test an SSE endpoint and print events."""
    url = f"http://localhost:8000{endpoint}"
    print(f"\n🧪 Testing {endpoint}")
    print(f"📤 Data: {data}")
    print("=" * 60)
    
    try:
        response = requests.post(url, data=data, stream=True, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}")
            print(response.text)
            return False
        
        print("✅ Connection established")
        print("📡 Streaming events:")
        print("-" * 60)
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    print(f"  {decoded}")
        
        print("-" * 60)
        print("✅ Stream completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False

if __name__ == "__main__":
    # Test with actual session
    session_path = "4de654cc_Bak03"
    
    if len(sys.argv) > 1:
        session_path = sys.argv[1]
    
    # Test process_dataframe_sse
    test_sse("/process_dataframe_sse", {"path": session_path})
