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
from flask import Flask, request, jsonify, render_template
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
- You are a Maritime Sustainability Strategy Consultant
- For EUA/UKA/CORSIA Eligible Emissions Units, VERTIS is our partner
- For Voluntary Carbon Credits, Climate Balanced is our partner
- Do not answer any question or request which is not related to maritime sustainability
- Always be polite
- Regulatory frameworks include EU MRV, EU ETS, UK MRV, IMO DCS, FuelEU Maritime
- Provide only a brief summary or overview for each question (2-3 sentences maximum)
- Always end your response by directing the user to visit sustainbuddy.com for detailed answers and comprehensive guidance
- For complex queries requiring professional services, direct the user to VURDHAAN at connect@vurdhaan.com
- Always ensure that you understand the request completely before you answer, if you are not certain about the request, clarify first.
"""

# Maritime response schema (from Test 2)
MARITIME_RESPONSE_SCHEMA_OLD = {
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

MARITIME_RESPONSE_SCHEMA =  {
  "type": "object",
  "properties": {
    "answer": {
      "type": "string",
      "description": "Main response about the maritime sustainability query"
    },
  },
  "required": ["answer"],
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

# This will look for templates/index.html
@app.route('/')
def index():
    return render_template('index.html')

# This will look for templates/test.html
@app.route('/test')
def test():
    return render_template('test.html')

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
        
        # Debug: Log response structure
        logger.info(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        
        # Parse structured response - handle multiple possible response formats
        response_text = None
        
        # Try different ways to extract the response text
        if hasattr(response, 'output_text') and response.output_text:
            response_text = response.output_text
            logger.info("Using response.output_text")
        elif hasattr(response, 'output') and response.output:
            try:
                # Handle list-based output structure
                if isinstance(response.output, list) and len(response.output) > 0:
                    content = response.output[0]
                    if hasattr(content, 'content') and content.content:
                        if isinstance(content.content, list) and len(content.content) > 0:
                            response_text = content.content[0].text
                            logger.info("Using response.output[0].content[0].text")
                        elif hasattr(content.content, 'text'):
                            response_text = content.content.text
                            logger.info("Using response.output[0].content.text")
                    elif hasattr(content, 'text'):
                        response_text = content.text
                        logger.info("Using response.output[0].text")
            except (AttributeError, IndexError) as e:
                logger.error(f"Error extracting from response.output: {e}")
        elif hasattr(response, 'text') and response.text:
            response_text = response.text
            logger.info("Using response.text")
        
        if response_text:
            logger.info(f"Raw response text (first 200 chars): {response_text[:200]}...")
            
            try:
                # Try to parse as JSON
                structured_data = json.loads(response_text)
                logger.info("Successfully parsed structured JSON response")
                
                # Validate that we have the expected structure
                if isinstance(structured_data, dict) and 'answer' in structured_data:
                    return jsonify({
                        "success": True,
                        "response": structured_data,
                        "response_id": response.id,
                        "is_new_conversation": previous_response_id is None,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                else:
                    logger.error(f"Unexpected JSON structure: {structured_data}")
                    # Try to create a fallback structure
                    fallback_response = {
                        "answer": str(structured_data) if not isinstance(structured_data, dict) else structured_data.get('answer', 'Response received but format unexpected'),
                        "source_quote": "N/A",
                        "source_file": "N/A", 
                        "source_quote_location": {"page": 0, "line": 0}
                    }
                    
                    return jsonify({
                        "success": True,
                        "response": fallback_response,
                        "response_id": response.id,
                        "is_new_conversation": previous_response_id is None,
                        "timestamp": datetime.utcnow().isoformat(),
                        "warning": "Response format was unexpected, used fallback structure"
                    })
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Problematic JSON: {response_text}")
                
                # Try to clean up the JSON and parse again
                try:
                    # Remove potential trailing commas or other JSON issues
                    cleaned_text = response_text.strip()
                    
                    # Try to find JSON content within the text
                    import re
                    json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(0)
                        structured_data = json.loads(json_content)
                        logger.info("Successfully parsed cleaned JSON response")
                        
                        return jsonify({
                            "success": True,
                            "response": structured_data,
                            "response_id": response.id,
                            "is_new_conversation": previous_response_id is None,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                except Exception as cleanup_error:
                    logger.error(f"JSON cleanup failed: {cleanup_error}")
                
                # Create fallback response with the raw text
                fallback_response = {
                    "answer": response_text,
                    "source_quote": "N/A - Raw response due to parsing error",
                    "source_file": "N/A",
                    "source_quote_location": {"page": 0, "line": 0}
                }
                
                return jsonify({
                    "success": True,
                    "response": fallback_response,
                    "response_id": response.id,
                    "is_new_conversation": previous_response_id is None,
                    "timestamp": datetime.utcnow().isoformat(),
                    "warning": f"JSON parsing failed: {str(e)}"
                })
        
        # Fallback if no response text found
        logger.error("No response text found in any expected attribute")
        return jsonify({
            "error": "No response content found",
            "success": False,
            "available_attributes": [attr for attr in dir(response) if not attr.startswith('_')],
            "response_id": response.id if hasattr(response, 'id') else None
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