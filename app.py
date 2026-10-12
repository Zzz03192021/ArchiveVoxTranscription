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
import torch

app = Flask(__name__)

# Suppress stderr
sys.stderr = io.StringIO()

# Set cache directory
cache_dir = '/tmp/.whisper_cache'
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['XDG_CACHE_HOME'] = cache_dir

# Device detection
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CPU count: {os.cpu_count()}")

# Load model at startup
print("Loading Whisper tiny model...")
model = whisper.load_model("tiny")
print(f"Model loaded successfully on device: {model.device}")
print(f"Whisper device: {model.device}")

# Queue system
job_queue = queue.Queue()
jobs = {}  # job_id -> {status, result, error, queue_position}
queue_lock = threading.Lock()
worker_threads = []

def process_queue(worker_id):
    """Background worker to process transcription jobs"""
    print(f"Worker {worker_id} thread started")
    while True:
        try:
            print(f"[Worker {worker_id}] Waiting for job... Queue size: {job_queue.qsize()}")
            job_id, temp_file_path, language = job_queue.get(timeout=5)  # Add timeout to detect if thread is stuck
            print(f"[Worker {worker_id}] Received job {job_id[:8]}")

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
                    fp16=False,  # Use full precision for CPU (fp16 is slower on CPU)
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
                import traceback
                traceback.print_exc()
                # Cleanup temp file on error
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

                with queue_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['error'] = str(e)

            job_queue.task_done()

        except queue.Empty:
            # Timeout is expected if no jobs in queue
            continue
        except Exception as e:
            print(f"[Worker {worker_id}] Queue worker error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

# Start queue workers (1 worker for CPU-only environment to avoid resource contention)
NUM_WORKERS = 1  # CPU-only: 1 worker to avoid vCPU contention
for i in range(NUM_WORKERS):
    queue_thread = threading.Thread(target=process_queue, args=(i,), daemon=False)
    queue_thread.start()
    worker_threads.append(queue_thread)
    print(f"Started worker thread {i+1}/{NUM_WORKERS}")

# Worker monitoring - restart crashed workers
def monitor_workers():
    while True:
        time.sleep(10)
        active_workers = sum(1 for t in worker_threads if t.is_alive())
        print(f"[Monitor] Active workers: {active_workers}/{NUM_WORKERS}")
        if active_workers < NUM_WORKERS:
            print(f"[Monitor] Restarting workers...")
            for i in range(NUM_WORKERS):
                if not worker_threads[i].is_alive():
                    print(f"[Monitor] Restarting worker {i+1}")
                    new_thread = threading.Thread(target=process_queue, args=(i,), daemon=False)
                    new_thread.start()
                    worker_threads[i] = new_thread

monitor_thread = threading.Thread(target=monitor_workers, daemon=False)
monitor_thread.start()
print("Started worker monitor thread")

# Keep-alive thread to prevent Railway from killing the service
def keepalive_thread_func():
    while True:
        try:
            time.sleep(30)  # Ping every 30 seconds
            # Just by running this loop, the thread keeps the process alive
            print(f"[Keepalive] Service alive at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            print(f"[Keepalive] Error: {e}")
            time.sleep(30)

keepalive_thread = threading.Thread(target=keepalive_thread_func, daemon=False)
keepalive_thread.start()
print("Started keepalive thread")

@app.route('/health', methods=['GET'])
def health():
    active_workers = sum(1 for t in worker_threads if t.is_alive())
    return jsonify({
        "status": "healthy",
        "model": "tiny",
        "device": str(model.device),
        "queue_size": job_queue.qsize(),
        "active_jobs": len(jobs),
        "active_workers": active_workers,
        "total_workers": NUM_WORKERS,
        "cuda_available": torch.cuda.is_available()
    })

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
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
