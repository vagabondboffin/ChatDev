#!/usr/bin/env python3
import json
import os
import glob
from datetime import datetime


def analyze_batch_results(batch_summary_file):
    """Analyze batch results for research insights"""

    with open(batch_summary_file, 'r') as f:
        data = json.load(f)

    results = data["results"]
    batch_info = data["batch_info"]

    print("📈 BATCH RESULTS ANALYSIS")
    print("=" * 50)

    # Basic statistics
    total_tasks = len(results)
    success_count = sum(1 for r in results if r["status"] == "success")
    success_rate = (success_count / total_tasks) * 100

    print(f"Total Tasks: {total_tasks}")
    print(f"Successful: {success_count} ({success_rate:.1f}%)")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Timeouts: {sum(1 for r in results if r['status'] == 'timeout')}")
    print(f"Errors: {sum(1 for r in results if r['status'] == 'error')}")

    # Duration analysis
    durations = [r["duration_seconds"] for r in results if r["duration_seconds"] > 0]
    if durations:
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        print(f"\n⏱️  Duration Analysis:")
        print(f"  Average: {avg_duration:.2f}s")
        print(f"  Max: {max_duration:.2f}s")
        print(f"  Min: {min_duration:.2f}s")

    # Project-specific results
    print(f"\n📋 Project Results:")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"  {status_icon} {result['project_name']}: {result['status']} ({result['duration_seconds']}s)")

    # Save analysis for research
    analysis_file = batch_summary_file.replace("_summary.json", "_analysis.json")
    analysis_data = {
        "analysis_timestamp": datetime.now().isoformat(),
        "summary_file": batch_summary_file,
        "statistics": {
            "total_tasks": total_tasks,
            "success_count": success_count,
            "success_rate": success_rate,
            "average_duration_seconds": avg_duration if durations else 0,
            "max_duration_seconds": max_duration if durations else 0,
            "min_duration_seconds": min_duration if durations else 0
        },
        "detailed_results": results
    }

    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)

    print(f"\n📊 Analysis saved to: {analysis_file}")
    return analysis_data


def find_latest_batch():
    """Find the most recent batch results"""
    batch_dirs = glob.glob("batch_runs/batch_*")
    if not batch_dirs:
        print("No batch results found")
        return None

    latest_batch = max(batch_dirs, key=os.path.getmtime)
    summary_file = os.path.join(latest_batch, "batch_summary.json")

    if os.path.exists(summary_file):
        return summary_file
    else:
        print(f"No summary file found in {latest_batch}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze batch task results")
    parser.add_argument("--summary-file", type=str, help="Path to batch summary file")
    parser.add_argument("--latest", action="store_true", help="Analyze latest batch")

    args = parser.parse_args()

    if args.latest:
        summary_file = find_latest_batch()
        if not summary_file:
            return
    elif args.summary_file:
        summary_file = args.summary_file
    else:
        print("Please specify --summary-file or --latest")
        return

    analyze_batch_results(summary_file)


if __name__ == "__main__":
    main()