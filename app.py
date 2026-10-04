from flask import Flask, request, jsonify
import whisper
import io
import sys
import os
import tempfile

app = Flask(__name__)

# Suppress stderr
sys.stderr = io.StringIO()

# Set cache directory
cache_dir = '/tmp/.whisper_cache'
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['XDG_CACHE_HOME'] = cache_dir

# Load model at startup
print("Loading Whisper tiny model...")
model = whisper.load_model("tiny")
print("Model loaded successfully")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "tiny"})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Get audio file
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'auto')
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            audio_file.save(temp_file.name)
            
            # Transcribe
            result = model.transcribe(
                temp_file.name,
                language=language if language != 'auto' else None,
                fp16=False,
                verbose=False
            )
            
            # Cleanup
            os.unlink(temp_file.name)
            
            return jsonify({
                "success": True,
                "text": result["text"],
                "language": language if language != 'auto' else result.get("language", "unknown"),
                "detected_language": result.get("language", "unknown")
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
