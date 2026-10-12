from flask import Flask, request, jsonify
import whisper
import io
import sys
import os
import tempfile
import threading
import queue
import uuid
import time

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

# Queue system
job_queue = queue.Queue()
jobs = {}  # job_id -> {status, result, error, queue_position}
queue_lock = threading.Lock()

def process_queue():
    """Background worker to process transcription jobs"""
    while True:
        try:
            job_id, temp_file_path, language = job_queue.get()

            start_time = time.time()
            queue_wait_time = start_time - jobs[job_id]['created_at']
            print(f"[Job {job_id[:8]}] Queue wait time: {queue_wait_time:.2f}s")

            with queue_lock:
                jobs[job_id]['status'] = 'processing'
                jobs[job_id]['queue_position'] = 0

            try:
                # Get audio file size for logging
                file_size = os.path.getsize(temp_file_path)
                print(f"[Job {job_id[:8]}] Audio file size: {file_size / 1024:.2f} KB")

                # Transcribe from saved file
                transcribe_start = time.time()
                result = model.transcribe(
                    temp_file_path,
                    language=language if language != 'auto' else None,
                    fp16=True,  # Enable half precision for faster inference
                    verbose=False
                )
                transcribe_time = time.time() - transcribe_start
                print(f"[Job {job_id[:8]}] Transcription time: {transcribe_time:.2f}s")

                # Cleanup temp file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

                total_time = time.time() - start_time
                print(f"[Job {job_id[:8]}] Total time: {total_time:.2f}s")

                with queue_lock:
                    jobs[job_id]['status'] = 'completed'
                    jobs[job_id]['result'] = {
                        "text": result["text"],
                        "language": language if language != 'auto' else result.get("language", "unknown"),
                        "detected_language": result.get("language", "unknown")
                    }

            except Exception as e:
                print(f"[Job {job_id[:8]}] Transcription error: {e}")
                # Cleanup temp file on error
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

                with queue_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['error'] = str(e)

            job_queue.task_done()

        except Exception as e:
            print(f"Queue worker error: {e}")
            time.sleep(1)

# Start multiple queue workers (2 workers for parallel processing)
NUM_WORKERS = 2
for i in range(NUM_WORKERS):
    queue_thread = threading.Thread(target=process_queue, daemon=True)
    queue_thread.start()
    print(f"Started worker thread {i+1}/{NUM_WORKERS}")

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

        # Save to temp file immediately
        temp_file_path = tempfile.mktemp(suffix='.webm')
        audio_file.save(temp_file_path)

        # Create job
        job_id = str(uuid.uuid4())

        with queue_lock:
            jobs[job_id] = {
                'status': 'queued',
                'queue_position': job_queue.qsize(),
                'result': None,
                'error': None,
                'created_at': time.time(),
                'temp_file': temp_file_path
            }

        print(f"[Job {job_id[:8]}] Job queued, position: {jobs[job_id]['queue_position']}")

        # Add to queue with file path instead of file object
        job_queue.put((job_id, temp_file_path, language))

        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "queue_position": jobs[job_id]['queue_position']
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    with queue_lock:
        if job_id not in jobs:
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        job = jobs[job_id]
        
        # Update queue position
        if job['status'] == 'queued':
            job['queue_position'] = job_queue.qsize()
        
        response = {
            "success": True,
            "job_id": job_id,
            "status": job['status'],
            "queue_position": job.get('queue_position', 0)
        }
        
        if job['status'] == 'completed':
            response['result'] = job['result']
        elif job['status'] == 'failed':
            response['error'] = job['error']
        
        return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
