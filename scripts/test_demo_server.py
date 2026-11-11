#!/usr/bin/env python3
"""
scripts/test_demo_server.py
Automated end-to-end test suite for the DBDD Demo Web Server & C++ Engine.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

def run_tests():
    port = 8899
    server_cmd = [sys.executable, "scripts/demo_server.py", "--port", str(port)]
    print(f"Starting demo server on port {port}...")
    server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Wait up to 3 seconds for server to start
        base_url = f"http://127.0.0.1:{port}"
        time.sleep(1.0)

        # 1. Test Static Index HTML
        print("1. Testing GET / (Static HTML)...")
        with urllib.request.urlopen(f"{base_url}/index.html", timeout=3) as res:
            assert res.status == 200
            content = res.read().decode("utf-8")
            assert "DBDD Cryptanalytic Framework" in content
            assert "presets-container" in content
        print("   -> PASS: Static HTML served correctly.")

        # 2. Test Health Endpoint
        print("2. Testing GET /api/health...")
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=3) as res:
            assert res.status == 200
            health_data = json.loads(res.read().decode("utf-8"))
            assert health_data["status"] == "ok"
            assert health_data["binary_ready"] is True
        print(f"   -> PASS: Health check OK, binary ready at {health_data['binary_path']}")

        # 3. Test Presets Endpoint
        print("3. Testing GET /api/presets...")
        with urllib.request.urlopen(f"{base_url}/api/presets", timeout=3) as res:
            assert res.status == 200
            presets = json.loads(res.read().decode("utf-8"))
            assert len(presets) == 5
            preset_ids = [p["id"] for p in presets]
            assert "toy_pipeline" in preset_ids
            assert "underdetermined" in preset_ids
            assert "overdetermined" in preset_ids
            assert "structural" in preset_ids
            assert "higher_dim" in preset_ids
        print(f"   -> PASS: All 5 presets returned ({preset_ids}).")

        # 4. Test POST /api/run with Toy Pipeline Preset
        print("4. Testing POST /api/run with Toy Pipeline...")
        toy_cfg = presets[0]["config"]
        req = urllib.request.Request(
            f"{base_url}/api/run",
            data=json.dumps(toy_cfg).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as res:
            assert res.status == 200
            run_data = json.loads(res.read().decode("utf-8"))
            assert "steps" in run_data
            assert len(run_data["steps"]) >= 5
            assert run_data["result"]["status"] == "SUCCESS"
            assert run_data["result"]["exact_match"] is True
            print(f"   -> Steps executed: {len(run_data['steps'])}")
            print(f"   -> Secret recovered: {run_data['result']['recovered_secret']}")
        print("   -> PASS: Pipeline executed and 100% secret recovered.")

        # 5. Test POST /api/run with Underdetermined (m < n) Preset
        print("5. Testing POST /api/run with Underdetermined LWE (n=5, m=3)...")
        under_cfg = presets[1]["config"]
        req2 = urllib.request.Request(
            f"{base_url}/api/run",
            data=json.dumps(under_cfg).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req2, timeout=10) as res:
            assert res.status == 200
            run_data2 = json.loads(res.read().decode("utf-8"))
            assert run_data2["result"]["status"] == "SUCCESS"
            assert run_data2["result"]["exact_match"] is True
            print(f"   -> Recovered: {run_data2['result']['recovered_secret']}")
        print("   -> PASS: Underdetermined instance solved via side hints.")

        print("\nALL SERVER & ENGINE INTEGRATION TESTS PASSED SUCCESSFULLY! (5/5)\n")

    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server_proc.kill()

if __name__ == "__main__":
    run_tests()

