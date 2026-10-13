import http.server
import socketserver
import json
import requests
import os
import threading
import sys
import io

# Force UTF-8 stdout/stderr so emoji in print() don't crash on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from local_patient_model import local_patient_engine
from leak_detector import leak_detector
from feedback_collector import save_feedback, get_feedback_count, get_feedback_log

PORT = 8000

class ProxyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.send_response(302)
            self.send_header('Location', '/lm_studio_arena.html')
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        if self.path == '/api/local_model/chat':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                req_data = json.loads(body.decode('utf-8'))

                # Extract input prompt
                user_input = req_data.get('input') or req_data.get('prompt') or ""
                if not user_input and req_data.get('messages'):
                    user_input = req_data['messages'][-1].get('content', '')
                system_prompt = req_data.get('system_prompt', '')

                response_text = local_patient_engine.generate_response(user_input, system_prompt)

                # ── Leak Detection ─────────────────────────────────────────
                leaks = leak_detector.detect(response_text)
                leak_detected = len(leaks) > 0
                leak_info = []

                if leak_detected:
                    print(leak_detector.describe_leaks(leaks))
                    log = save_feedback(user_input, response_text, leaks)
                    leak_info = [
                        {
                            "field":      l["field"],
                            "patient_id": l["patient_id"],
                            "value":      l["value"],
                        }
                        for l in leaks
                    ]
                # ───────────────────────────────────────────────────────────

                res_body = json.dumps({
                    "model": "local-patient-model-python",
                    "output": [
                        {
                            "type":    "message",
                            "content": response_text
                        }
                    ],
                    "choices": [
                        {
                            "message": {
                                "role":    "assistant",
                                "content": response_text
                            }
                        }
                    ],
                    "leak_detected":      leak_detected,
                    "leak_info":          leak_info,
                    "feedback_total":     get_feedback_count(),
                })

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(res_body.encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                err_msg = json.dumps({"error": str(e)})
                self.wfile.write(err_msg.encode('utf-8'))

        elif self.path == '/api/proxy':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                req_data = json.loads(body.decode('utf-8'))

                target_url = req_data.get('target_url')
                payload = req_data.get('payload', {})

                if not target_url:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error": "Missing target_url"}')
                    return

                # Forward request to target endpoint
                resp = requests.post(
                    target_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=120
                )

                self.send_response(resp.status_code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(resp.content)

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                err_msg = json.dumps({"error": str(e)})
                self.wfile.write(err_msg.encode('utf-8'))

        elif self.path == '/api/fetch_models':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length) if content_length > 0 else b'{}'
                req_data = json.loads(body.decode('utf-8')) if body else {}

                base_url = req_data.get('base_url', 'http://localhost:1234/v1').rstrip('/')
                models_url = f"{base_url}/models" if not base_url.endswith('/models') else base_url

                resp = requests.get(models_url, timeout=5)
                self.send_response(resp.status_code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(resp.content)
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                err_msg = json.dumps({"error": str(e)})
                self.wfile.write(err_msg.encode('utf-8'))
        elif self.path == '/api/retrain':
            # Trigger LoRA retraining from feedback examples in a background thread
            try:
                from retrain_from_feedback import retrain
                def run_retrain():
                    result = retrain()
                    print(f"[Retrain] Done: {result}")
                t = threading.Thread(target=run_retrain, daemon=True)
                t.start()
                res_body = json.dumps({
                    "status":  "started",
                    "message": "Retraining started in background. Check server console for progress.",
                    "feedback_examples": get_feedback_count(),
                })
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(res_body.encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif self.path == '/api/feedback_status':
            # Return how many feedback examples have been collected
            try:
                log = get_feedback_log()
                res_body = json.dumps({
                    "feedback_count": get_feedback_count(),
                    "log": log[-10:],   # last 10 entries
                })
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(res_body.encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()


def run_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ProxyHTTPRequestHandler) as httpd:
        print(f"Proxy Server running at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == '__main__':
    run_server()
