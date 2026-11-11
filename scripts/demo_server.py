#!/usr/bin/env python3
"""
scripts/demo_server.py

Zero-dependency local HTTP server for the DBDD / LWE Side-Information Web Demo.
Serves static assets from web/ and bridges frontend requests to the C++ dbdd_demo_runner binary.
"""

import argparse
import http.server
import json
import os
import subprocess
import sys
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
BUILD_DIR = ROOT_DIR / "build"
BINARY_NAME = "dbdd_demo_runner"

PRESETS_DATA = [
    {
        "id": "toy_pipeline",
        "name": "1. Toy Pipeline Benchmark",
        "badge": "n=3, m=3, q=1051",
        "description": "Exact match with examples/01_toy_pipeline.cpp. Includes short-vector q-vector projection, coordinate s_0 perfect hint, and noisy s_1 approximate hint. Demonstrates 100% reliable BKZ-2 secret recovery.",
        "config": {
            "n": 3,
            "m": 3,
            "q": "1051",
            "dist": "ternary",
            "sigma": 1.0,
            "seed": 53,
            "bkz_blocksize": 2,
            "bkz_tours": 15,
            "hints": [
                {
                    "type": "short_vector",
                    "label": "Short-Vector Hint: q-vector projection",
                    "vector": [1051, 0, 0, 0, 0, 0, 0]
                },
                {
                    "type": "perfect",
                    "label": "Perfect Hint: Exact s_0 revealed",
                    "vector": [1, 0, 0]
                },
                {
                    "type": "approximate",
                    "label": "Approximate Hint: s_1 with noise",
                    "vector": [0, 1, 0],
                    "variance": 0.50
                }
            ]
        }
    },
    {
        "id": "underdetermined",
        "name": "2. Underdetermined LWE (Few Samples)",
        "badge": "n=5, m=3 (m < n), q=257",
        "description": "Rectangular LWE with m < n information deficit. Unsolvable without side hints. Solved via two perfect coordinate hints (s_0, s_1) and a modular hint s_2 (mod 2).",
        "config": {
            "n": 5,
            "m": 3,
            "q": "257",
            "dist": "ternary",
            "sigma": 1.0,
            "seed": 101,
            "bkz_blocksize": 2,
            "bkz_tours": 15,
            "hints": [
                {
                    "type": "perfect",
                    "label": "Perfect Hint: Coordinate s_0",
                    "vector": [1, 0, 0, 0, 0]
                },
                {
                    "type": "perfect",
                    "label": "Perfect Hint: Coordinate s_1",
                    "vector": [0, 1, 0, 0, 0]
                },
                {
                    "type": "modular",
                    "label": "Modular Hint: Coordinate s_2 (mod 2)",
                    "vector": [0, 0, 1, 0, 0],
                    "modulus": 2
                }
            ]
        }
    },
    {
        "id": "overdetermined",
        "name": "3. Overdetermined LWE with Modular Leakage",
        "badge": "n=3, m=6 (m > n), q=257",
        "description": "Overdetermined LWE with m > n and discrete Gaussian noise (sigma=1.2). Incorporates two modular hints modulo 3 and an approximate side-channel hint.",
        "config": {
            "n": 3,
            "m": 6,
            "q": "257",
            "dist": "gaussian",
            "sigma": 1.2,
            "seed": 202,
            "bkz_blocksize": 2,
            "bkz_tours": 15,
            "hints": [
                {
                    "type": "modular",
                    "label": "Modular Hint: Coordinate s_0 (mod 3)",
                    "vector": [1, 0, 0],
                    "modulus": 3
                },
                {
                    "type": "modular",
                    "label": "Modular Hint: Coordinate s_1 (mod 3)",
                    "vector": [0, 1, 0],
                    "modulus": 3
                },
                {
                    "type": "approximate",
                    "label": "Approximate Hint: Coordinate s_2 with noise",
                    "vector": [0, 0, 1],
                    "variance": 0.35
                }
            ]
        }
    },
    {
        "id": "structural",
        "name": "4. Structural Balanced Secret",
        "badge": "n=6, m=4, q=521",
        "description": "Secret satisfies balanced structural constraint sum(s_i) = 0. Demonstrates integration of a full-vector linear constraint alongside difference leakage s_0 - s_1.",
        "config": {
            "n": 6,
            "m": 4,
            "q": "521",
            "dist": "ternary",
            "sigma": 1.0,
            "seed": 303,
            "bkz_blocksize": 2,
            "bkz_tours": 15,
            "hints": [
                {
                    "type": "perfect",
                    "label": "Structural Hint: sum(s_i) = 0",
                    "vector": [1, 1, 1, 1, 1, 1],
                    "scalar_value": 0.0
                },
                {
                    "type": "approximate",
                    "label": "Approximate Hint: Leakage on s_0 - s_1",
                    "vector": [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    "variance": 0.30
                }
            ]
        }
    },
    {
        "id": "higher_dim",
        "name": "5. Higher-Dimension Challenge",
        "badge": "n=10, m=12, q=1051",
        "description": "Larger 23-dimensional ambient lattice with 5 side hints (3 perfect, 2 approximate). Demonstrates smooth degradation of beta and fast BKZ-3 reduction.",
        "config": {
            "n": 10,
            "m": 12,
            "q": "1051",
            "dist": "ternary",
            "sigma": 1.0,
            "seed": 404,
            "bkz_blocksize": 3,
            "bkz_tours": 15,
            "hints": [
                {
                    "type": "perfect",
                    "label": "Perfect Hint: Coordinate s_0",
                    "vector": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                },
                {
                    "type": "perfect",
                    "label": "Perfect Hint: Coordinate s_1",
                    "vector": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                },
                {
                    "type": "perfect",
                    "label": "Perfect Hint: Coordinate s_2",
                    "vector": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                },
                {
                    "type": "approximate",
                    "label": "Approximate Hint: Coordinate s_3",
                    "vector": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    "variance": 0.25
                },
                {
                    "type": "approximate",
                    "label": "Approximate Hint: Coordinate s_4",
                    "vector": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    "variance": 0.25
                }
            ]
        }
    }
]

class DemoRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, binary_path=None, **kwargs):
        self.binary_path = binary_path
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/presets":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(PRESETS_DATA, indent=2).encode("utf-8"))
        elif self.path == "/api/health":
            ready = os.path.isfile(self.binary_path) and os.access(self.binary_path, os.X_OK)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            resp = {
                "status": "ok",
                "binary_ready": ready,
                "binary_path": str(self.binary_path)
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/run":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            
            try:
                # If preset ID is provided directly
                req_json = json.loads(body) if body else {}
                
                cmd = [str(self.binary_path), "--json", "-"]
                proc = subprocess.run(
                    cmd,
                    input=json.dumps(req_json),
                    text=True,
                    capture_output=True,
                    timeout=30
                )

                if proc.returncode != 0 and not proc.stdout:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    err = {
                        "error": "Engine error",
                        "stderr": proc.stderr,
                        "returncode": proc.returncode
                    }
                    self.wfile.write(json.dumps(err).encode("utf-8"))
                    return

                # Send stdout from dbdd_demo_runner directly
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(proc.stdout.encode("utf-8"))

            except subprocess.TimeoutExpired:
                self.send_response(504)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Pipeline execution timed out"}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def main():
    parser = argparse.ArgumentParser(description="DBDD Web Demo Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--bind", type=str, default="127.0.0.1", help="Address to bind to (default: 127.0.0.1)")
    parser.add_argument("--build-dir", type=str, default=str(BUILD_DIR), help="Build directory containing dbdd_demo_runner")
    args = parser.parse_args()

    binary_path = Path(args.build_dir) / BINARY_NAME
    if not binary_path.exists():
        print(f"[WARNING] Demo runner binary not found at {binary_path}. Please build it first with:")
        print("  ninja -C build dbdd_demo_runner")

    server_address = (args.bind, args.port)
    handler = lambda *h_args, **h_kwargs: DemoRequestHandler(*h_args, binary_path=binary_path, **h_kwargs)
    
    httpd = http.server.ThreadingHTTPServer(server_address, handler)
    print(f"=================================================================")
    print(f" DBDD Interactive Demo Server Running at http://{args.bind}:{args.port}")
    print(f" Serving web files from: {WEB_DIR}")
    print(f" Using C++ engine binary: {binary_path}")
    print(f" Press Ctrl+C to stop.")
    print(f"=================================================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()

if __name__ == "__main__":
    main()

