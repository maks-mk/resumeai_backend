import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-flash-lite-latest")

# Store chat history for context
chat_history = []

@app.route("/api/analyze", methods=["POST"])
def analyze_resume():
    """Endpoint to analyze resume and provide feedback"""
    try:
        data = request.json
        resume_text = data.get("resume_text", "")
        
        if not resume_text:
            return jsonify({"error": "No resume text provided"}), 400
        
        # Create a prompt for resume analysis
        analysis_prompt = f"""Analyze the following resume and provide constructive feedback on:
1. Overall structure and format
2. Content clarity and relevance
3. Achievements and metrics
4. Suggestions for improvement

Resume:
{resume_text}"""
        
        # Send to Gemini
        response = model.generate_content(analysis_prompt)
        
        return jsonify({
            "analysis": response.text,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """Endpoint for chatting about resume"""
    try:
        data = request.json
        user_message = data.get("message", "")
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Add to chat history
        chat_history.append({"role": "user", "content": user_message})
        
        # Create context from chat history
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        
        # Send to Gemini
        response = model.generate_content(context)
        
        # Add response to history
        chat_history.append({"role": "assistant", "content": response.text})
        
        return jsonify({
            "response": response.text,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(debug=True)
