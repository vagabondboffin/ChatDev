#!/usr/bin/env python3
import json
import os
import sys
import argparse
import time
from datetime import datetime
import subprocess


def load_tasks(json_file_path):
    """Load programming tasks from JSON file"""
    try:
        with open(json_file_path, 'r') as f:
            tasks = json.load(f)
        print(f"✅ Loaded {len(tasks)} tasks from {json_file_path}")
        return tasks
    except Exception as e:
        print(f"❌ Error loading tasks from {json_file_path}: {e}")
        return []


def setup_batch_environment():
    """Setup environment for batch processing"""
    # Create batch run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = f"batch_runs/batch_{timestamp}"
    os.makedirs(batch_dir, exist_ok=True)

    # Create summary file
    summary_file = os.path.join(batch_dir, "batch_summary.json")

    print(f"📁 Batch output directory: {batch_dir}")
    return batch_dir, summary_file


def run_single_task(task_description, project_name, model="llama2", batch_dir=None):
    """Run a single task with ChatDev and return results"""

    # Create task-specific directory
    if batch_dir:
        task_dir = os.path.join(batch_dir, project_name)
        os.makedirs(task_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🚀 Starting Task: {project_name}")
    print(f"📝 Description: {task_description}")
    print(f"🤖 Model: {model}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Run ChatDev for this task
        cmd = [
            sys.executable, "run_ollama_simple.py",
            "--task", task_description,
            "--name", project_name,
            "--model", model
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout per task
        )

        end_time = time.time()
        duration = end_time - start_time

        # Capture results
        task_result = {
            "project_name": project_name,
            "description": task_description,
            "status": "success" if result.returncode == 0 else "failed",
            "duration_seconds": round(duration, 2),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat()
        }

        if result.returncode == 0:
            print(f"✅ Task completed successfully in {duration:.2f}s")
        else:
            print(f"❌ Task failed after {duration:.2f}s")
            print(f"Error: {result.stderr}")

        return task_result

    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time

        print(f"⏰ Task timed out after {duration:.2f}s")

        return {
            "project_name": project_name,
            "description": task_description,
            "status": "timeout",
            "duration_seconds": round(duration, 2),
            "return_code": -1,
            "stdout": "",
            "stderr": "Task exceeded 30 minute timeout",
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat()
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"💥 Unexpected error: {e}")

        return {
            "project_name": project_name,
            "description": task_description,
            "status": "error",
            "duration_seconds": round(duration, 2),
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat()
        }


def run_batch_tasks(json_file_path, model="llama2", start_index=0, end_index=None):
    """Run all tasks from JSON file in batch mode"""

    # Load tasks
    tasks = load_tasks(json_file_path)
    if not tasks:
        return

    # Setup batch environment
    batch_dir, summary_file = setup_batch_environment()

    # Determine task range
    if end_index is None or end_index > len(tasks):
        end_index = len(tasks)

    tasks_to_run = tasks[start_index:end_index]

    print(f"🎯 Running tasks {start_index} to {end_index - 1} (total: {len(tasks_to_run)})")

    # Run each task
    results = []
    for i, task in enumerate(tasks_to_run, start=start_index):
        task_num = i + 1
        total_tasks = len(tasks_to_run)

        print(f"\n📋 Task {task_num}/{total_tasks} ({i}/{len(tasks) - 1})")

        # Run the task
        result = run_single_task(
            task_description=task["description"],
            project_name=task["project_name"],
            model=model,
            batch_dir=batch_dir
        )

        results.append(result)

        # Save incremental results
        with open(summary_file, 'w') as f:
            json.dump({
                "batch_info": {
                    "total_tasks": len(tasks_to_run),
                    "completed_tasks": len(results),
                    "batch_dir": batch_dir,
                    "model": model,
                    "start_time": datetime.now().isoformat()
                },
                "results": results
            }, f, indent=2)

        # Brief pause between tasks
        if i < len(tasks_to_run) - 1:
            print("⏳ Waiting 10 seconds before next task...")
            time.sleep(10)

    # Print final summary
    print(f"\n{'=' * 60}")
    print("📊 BATCH PROCESSING COMPLETE")
    print(f"{'=' * 60}")

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    error_count = sum(1 for r in results if r["status"] == "error")

    print(f"✅ Successful: {success_count}/{len(results)}")
    print(f"❌ Failed: {failed_count}/{len(results)}")
    print(f"⏰ Timeout: {timeout_count}/{len(results)}")
    print(f"💥 Errors: {error_count}/{len(results)}")
    print(f"📁 Results saved to: {summary_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ChatDev batch tasks from JSON file")
    parser.add_argument("--json-file", type=str, required=True, help="Path to JSON file with tasks")
    parser.add_argument("--model", type=str, default="llama2", help="Ollama model to use")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--end", type=int, help="End index (exclusive)")

    args = parser.parse_args()

    # Verify JSON file exists
    if not os.path.exists(args.json_file):
        print(f"❌ JSON file not found: {args.json_file}")
        return

    # Run batch tasks
    run_batch_tasks(
        json_file_path=args.json_file,
        model=args.model,
        start_index=args.start,
        end_index=args.end
    )


if __name__ == "__main__":
    main()