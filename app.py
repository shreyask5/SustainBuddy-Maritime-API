#!/usr/bin/env python3
"""
Maritime Sustainability Chatbot Backend using OpenAI Responses API with Vector Store
Features:
- Conversation state management using previous_response_id
- Structured JSON responses with source citations
- Vector store integration for maritime documents
- All queries assumed to be maritime-related
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Maritime sustainability instructions
MARITIME_INSTRUCTIONS = """
You are a Maritime Sustainability Strategy Consultant with the following guidelines:

- For EUA/UKA/CORSIA Eligible Emissions Units, VERTIS is our partner
- For Voluntary Carbon Credits, Climate Balanced is our partner
- Do not answer any question or request which is not related to maritime sustainability
- Always be polite and professional
- Regulatory frameworks include EU MRV, EU ETS, UK MRV, IMO DCS, FuelEU Maritime
- For complex queries requiring professional services, direct the user to VURDHAAN at connect@vurdhaan.com
- Always ensure that you understand the request completely before you answer, if you are not certain about the request, clarify first
- Use the file search tool to find relevant information from maritime documents and provide comprehensive, accurate answers
"""

# Maritime response schema (from Test 2)
MARITIME_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Main response about the maritime sustainability query"
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

def validate_environment():
    """Validate required environment variables"""
    api_key = os.environ.get("OPENAI_API_KEY")
    vector_store_id = os.environ.get("VECTOR_STORE_ID")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    if not vector_store_id or vector_store_id == 'your-vector-store-id-here':
        raise ValueError("VECTOR_STORE_ID environment variable not set or using placeholder value")
    
    return api_key, vector_store_id

# Removed maritime keyword check as all queries are maritime-related

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Maritime Sustainability Chatbot"
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint using OpenAI Responses API with conversation state and vector store"""
    try:
        # Validate environment
        api_key, vector_store_id = validate_environment()
        
        # Get request data
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Missing 'message' in request body",
                "success": False
            }), 400
        
        user_message = data['message'].strip()
        previous_response_id = data.get('previous_response_id')  # Optional for conversation continuity
        
        if not user_message:
            return jsonify({
                "error": "Empty message provided",
                "success": False
            }), 400
        
        logger.info(f"Processing query: {user_message[:100]}...")
        if previous_response_id:
            logger.info(f"Continuing conversation from response ID: {previous_response_id}")
        
        # Prepare the API call parameters
        api_params = {
            "model": "gpt-4o-mini",
            "tools": [{
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "maritime_response",
                    "schema": MARITIME_RESPONSE_SCHEMA,
                    "strict": True
                }
            }
        }
        
        # Handle conversation state
        if previous_response_id:
            # Continuing conversation - use previous_response_id and format input as messages
            api_params["previous_response_id"] = previous_response_id
            api_params["input"] = [{"role": "user", "content": user_message}]
        else:
            # New conversation - include instructions and use string input
            api_params["instructions"] = MARITIME_INSTRUCTIONS
            api_params["input"] = user_message
        
        # Call OpenAI Responses API with conversation state + file_search
        response = client.responses.create(**api_params)
        
        logger.info(f"OpenAI Response ID: {response.id}")
        
        # Parse structured response
        if hasattr(response, 'output_text'):
            try:
                structured_data = json.loads(response.output_text)
                logger.info("Successfully parsed structured JSON response")
                
                return jsonify({
                    "success": True,
                    "response": structured_data,
                    "response_id": response.id,
                    "is_new_conversation": previous_response_id is None,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return jsonify({
                    "error": "Failed to parse AI response",
                    "success": False,
                    "raw_response": response.output_text[:200] + "..." if len(response.output_text) > 200 else response.output_text
                }), 500
        
        # Handle alternative response structure
        elif hasattr(response, 'output') and response.output:
            try:
                content = response.output[0]
                if hasattr(content, 'content') and content.content:
                    text = content.content[0].text
                    structured_data = json.loads(text)
                    
                    return jsonify({
                        "success": True,
                        "response": structured_data,
                        "response_id": response.id,
                        "is_new_conversation": previous_response_id is None,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            except (json.JSONDecodeError, AttributeError, IndexError) as e:
                logger.error(f"Failed to parse alternative response format: {e}")
        
        # Fallback if response structure is unexpected
        return jsonify({
            "error": "Unexpected response format from AI",
            "success": False,
            "available_attributes": [attr for attr in dir(response) if not attr.startswith('_')]
        }), 500
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        return jsonify({
            "error": "Internal server error occurred",
            "success": False,
            "details": str(e) if app.debug else None
        }), 500

@app.route('/new-conversation', methods=['POST'])
def new_conversation():
    """Start a new conversation (convenience endpoint for frontend)"""
    return jsonify({
        "success": True,
        "message": "Ready for new conversation. Send your first message to /chat without previous_response_id.",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/vector-store-info', methods=['GET'])
def vector_store_info():
    """Get information about the vector store"""
    try:
        vector_store_id = os.environ.get("VECTOR_STORE_ID")
        
        if not vector_store_id:
            return jsonify({
                "error": "Vector store ID not configured",
                "success": False
            }), 500
        
        # Check which API to use
        if hasattr(client, 'vector_stores'):
            vs_client = client.vector_stores
            api_type = "direct"
        elif hasattr(client, 'beta') and hasattr(client.beta, 'vector_stores'):
            vs_client = client.beta.vector_stores
            api_type = "beta"
        else:
            return jsonify({
                "error": "Vector stores not supported in this OpenAI client version",
                "success": False
            }), 500
        
        # Get vector store details
        vector_store = vs_client.retrieve(vector_store_id)
        
        return jsonify({
            "success": True,
            "vector_store": {
                "id": vector_store_id,
                "name": vector_store.name,
                "status": vector_store.status,
                "api_type": api_type,
                "file_counts": {
                    "total": getattr(vector_store.file_counts, 'total', 'N/A'),
                    "completed": getattr(vector_store.file_counts, 'completed', 'N/A')
                } if hasattr(vector_store, 'file_counts') else None
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting vector store info: {e}")
        return jsonify({
            "error": "Failed to retrieve vector store information",
            "success": False,
            "details": str(e) if app.debug else None
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "success": False
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "success": False
    }), 405

if __name__ == '__main__':
    # Validate environment on startup
    try:
        validate_environment()
        logger.info("Environment validation successful")
        logger.info("Maritime Sustainability Chatbot Backend starting...")
        
        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=os.environ.get('FLASK_ENV') == 'development'
        )
        
    except ValueError as e:
        logger.error(f"Startup failed: {e}")
        print(f"❌ Configuration Error: {e}")
        print("\nPlease ensure your .env file contains:")
        print("OPENAI_API_KEY=your_actual_api_key_here")
        print("VECTOR_STORE_ID=your_actual_vector_store_id_here")
        exit(1)