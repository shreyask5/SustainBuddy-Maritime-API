#!/usr/bin/env python3
"""
Simple test for Responses API with vector store - using the exact format from user's examples
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def test_responses_api_with_vector_store():
    """Test the Responses API with vector store using the exact format from user examples"""
    
    print("🧪 Testing Responses API with Vector Store - Correct Format")
    print("=" * 60)
    
    vector_store_id = os.getenv('VECTOR_STORE_ID')
    
    if not vector_store_id or vector_store_id == 'your-vector-store-id-here':
        print("❌ No vector store ID configured. Please set VECTOR_STORE_ID in your .env file")
        return
    
    print(f"📁 Vector Store ID: {vector_store_id}")
    print(f"🤖 Model: gpt-4o-mini")
    
    # Test 1: Basic Responses API with file_search (user's exact format)
    print("\n" + "─" * 50)
    print("🧪 Test 1: Basic file_search with vector store")
    print("─" * 50)
    
    try:
        query = "What are the main environmental challenges facing the maritime industry?"
        instructions = """
        You are a maritime sustainability expert. Use the file search tool to find relevant information 
        from the maritime documents and provide a comprehensive answer about environmental challenges.
        """
        
        print(f"📝 Query: {query}")
        print("🚀 Calling Responses API...")
        
        # Using the EXACT format from user's examples
        response = client.responses.create(
            input=query,
            instructions=instructions,
            model="o4-mini",
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }]
        )
        
        print("✅ Success! Response received")
        print(f"📋 Response ID: {response.id}")
        
        # Extract response text
        if hasattr(response, 'output_text'):
            print(f"📄 Response length: {len(response.output_text)} characters")
            print(f"📄 Response: {response.output_text[:300]}...")
        elif hasattr(response, 'output'):
            print("📄 Found response.output, extracting content...")
            if response.output and len(response.output) > 0:
                content = response.output[0]
                if hasattr(content, 'content') and content.content:
                    text = content.content[0].text
                    print(f"📄 Response length: {len(text)} characters")
                    print(f"📄 Response: {text[:300]}...")
        else:
            print("⚠️  Could not extract response text")
            print(f"Available attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        print(f"Error type: {type(e)}")
        
        if hasattr(e, 'response'):
            print(f"HTTP Status: {getattr(e.response, 'status_code', 'N/A')}")
            try:
                error_text = getattr(e.response, 'text', '')
                if error_text:
                    error_data = json.loads(error_text)
                    print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw error: {getattr(e.response, 'text', '')[:200]}...")
    
    # Test 2: Responses API with structured output + vector store
    print("\n" + "─" * 50)
    print("🧪 Test 2: Structured output + file_search")
    print("─" * 50)
    
    try:
        query = "How can shipping companies reduce carbon emissions?"
        instructions = """
        You are a maritime sustainability expert. Use the file search tool to find relevant information 
        and provide a structured response about carbon emission reduction strategies.
        """
        
        # Maritime response schema
        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Main response about carbon emission reduction"
                },
                "source_quote": {
                    "type": "string", 
                    "description": "Relevant quote from source documents"
                },
                "source_file": {
                    "type": "string",
                    "description": "Name of the source file"
                },
                "source_quote_location": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "line": {"type": "integer"}
                    },
                    "required": ["page", "line"],
                    "additionalProperties": False
                }
            },
            "required": ["answer", "source_quote", "source_file", "source_quote_location"],
            "additionalProperties": False
        }
        
        print(f"📝 Query: {query}")
        print("🚀 Calling Responses API with structured output...")
        
        response = client.responses.create(
            input=query,
            instructions=instructions,
            model="o4-mini",
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "maritime_response",
                    "schema": schema,
                    "strict": True
                }
            }
        )
        
        print("✅ Success! Structured response received")
        print(f"📋 Response ID: {response.id}")
        
        # Parse structured response
        if hasattr(response, 'output_text'):
            try:
                structured_data = json.loads(response.output_text)
                print("🎯 Successfully parsed structured JSON!")
                print(json.dumps(structured_data, indent=2))
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                print(f"Raw response: {response.output_text[:200]}...")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        print(f"Error type: {type(e)}")

def test_vector_store_info():
    """Get basic info about the vector store"""
    
    print("\n" + "─" * 50)
    print("🔍 Vector Store Information")
    print("─" * 50)
    
    vector_store_id = os.getenv('VECTOR_STORE_ID')
    
    try:
        # Check which API to use
        if hasattr(client, 'vector_stores'):
            vs_client = client.vector_stores
            api_type = "direct"
        elif hasattr(client, 'beta') and hasattr(client.beta, 'vector_stores'):
            vs_client = client.beta.vector_stores
            api_type = "beta"
        else:
            print("❌ Vector stores not supported")
            return
        
        print(f"Using {api_type} vector stores API")
        
        # Get vector store details
        vector_store = vs_client.retrieve(vector_store_id)
        print(f"✅ Vector Store: {vector_store.name}")
        print(f"   Status: {vector_store.status}")
        print(f"   Total files: {vector_store.file_counts.total if hasattr(vector_store.file_counts, 'total') else 'N/A'}")
        print(f"   Completed files: {vector_store.file_counts.completed if hasattr(vector_store.file_counts, 'completed') else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Failed to get vector store info: {e}")

if __name__ == "__main__":
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        exit(1)
    
    print("🔑 API Key found")
    
    # Test vector store info
    test_vector_store_info()
    
    # Test Responses API
    test_responses_api_with_vector_store()
    
    print("\n" + "=" * 60)
    print("🎉 Testing complete!")
    print("\nIf tests pass, your Flask app should work correctly with:")
    print("1. Vector store file search integration")
    print("2. Structured JSON output")
    print("3. Proper source citations from your maritime documents")