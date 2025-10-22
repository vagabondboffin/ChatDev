#!/usr/bin/env python3
import os
import sys
import subprocess
import time


def start_visualizer():
    """Start the ChatDev visualizer Flask app"""
    visualizer_path = os.path.join(os.path.dirname(__file__), "visualizer", "app.py")

    if os.path.exists(visualizer_path):
        print("Starting ChatDev Visualizer...")
        print("The visualizer will be available at: http://localhost:5000")
        print("Press Ctrl+C to stop the visualizer")

        try:
            # Start the Flask app
            subprocess.run([sys.executable, visualizer_path])
        except KeyboardInterrupt:
            print("\nVisualizer stopped")
    else:
        print(f"Visualizer not found at: {visualizer_path}")
        print("You can still view the logs in the 'logs/' directory")


if __name__ == "__main__":
    start_visualizer()