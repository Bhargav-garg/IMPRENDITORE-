"""
Launcher for Autonomous Smart Traffic Management System.
Starts the Flask server and opens the dashboard in the default browser.
"""
import os
import sys
import webbrowser
import threading
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def open_browser():
    """Open dashboard in default browser after a short delay."""
    time.sleep(2.5)
    webbrowser.open("http://127.0.0.1:5000")


def main():
    print("=" * 70)
    print("   AUTONOMOUS SMART TRAFFIC MANAGEMENT SYSTEM")
    print("   Cities: Bhopal | Gwalior | New Delhi | Bangalore | Chandigarh")
    print("   Deep Learning: YOLO + CNN + LSTM")
    print("   Quantum Engine: Qiskit QAOA (5-qubit)")
    print("   Dashboard: http://127.0.0.1:5000")
    print("=" * 70)

    threading.Thread(target=open_browser, daemon=True).start()

    from backend.app import create_app
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
