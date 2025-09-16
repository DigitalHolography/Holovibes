import subprocess
import threading
import sys
import os
import shutil
import argparse
import pandas as pd

def benchmark():
    if len(sys.argv) < 1:
        print("Usage: ./script.py <input_file.holo>")
        sys.exit(1)

    if os.path.exists(sys.argv[1]) == False and sys.argv[1].endswith(".holo"):
        print("Input file does not exist or is not a .holo file")
        sys.exit(1)

    if os.path.exists("build/bin/Holovibes.exe") == False:
        print("Holovibes executable not found, please build it first.")
        sys.exit(1)

    def run_holovibes():
        os.makedirs("benchmark", exist_ok=True)

        cmd = ["nsys", "profile", "--stats=true", "--output=benchmark/benchmark_report", "build/bin/Holovibes.exe"]

        if args.input:
            cmd.append("--input", args.input)
        
        process = subprocess.Popen(cmd)
        process.wait()
        process = subprocess.Popen(["nsys", "stats", "benchmark/benchmark_report.nsys-rep", "--format=csv", "--output=benchmark/benchmark_report", "--force-overwrite=true", "--force-export=true"])
        process.wait()
        os.remove("benchmark/benchmark_report.nsys-rep",)
        os.remove("benchmark/benchmark_report.sqlite",)

    thread = threading.Thread(target=run_holovibes)
    thread.start()

    thread.join()

    print("Holovibes benchmark terminated.")

def preview(csv_dir="benchmark"):
    def list_csv_files(directory):
        return [f for f in os.listdir(directory) if f.endswith(".csv")]

    def view_csv_file(filepath):
        try:
            df = pd.read_csv(filepath)
            print(f"\n📄 File: {filepath}")
            print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            print(f"Column names: {list(df.columns)}")
            if df.empty:
                print("⚠️ File is empty.")
            else:
                print("\nPreview:")
                print(df.head(args.nrows).to_string(index=False))
        except Exception as e:
             print(f"❌ Error reading {filepath}: {e}")

    files = list_csv_files(csv_dir)
    if not files:
        print("No CSV files found.")
        return

    print("Available CSV files:\n")
    for i, f in enumerate(files, 1):
        size = os.path.getsize(os.path.join(csv_dir, f))
        print(f"[{i}] {f} ({size} bytes)")

    while True:
        choice = input("\nEnter the number of the CSV to view (Enter to quit): ")
        if not choice:
            break
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        idx = int(choice) - 1
        if 0 <= idx < len(files):
            filepath = os.path.join(csv_dir, files[idx])
            view_csv_file(filepath, args.nrows)
        else:
            print("Invalid choice.")

def main():
    parser = argparse.ArgumentParser(description="Benchmarking and CSV report Viewing Tool")
    subparsers = parser.add_subparsers(dest="command")

    bench_parsers = subparsers.add_parser("benchmark", help="Run Holovibes benchmark")
    preview_parsers = subparsers.add_parser("preview", help="View CSV report files interactively")
    subparsers.add_parser("clean", help="Clean the benchmark directory")

    bench_parsers.add_argument("--input", type=str, help="Input file for Holovibes")
    preview_parsers.add_argument("--nrows", type=int, help="Number of rows to preview from CSV files (default: 10)")

    global args 
    args = parser.parse_args()

    if args.command == "benchmark":
        benchmark()
    elif args.command == "preview":
        preview()
    elif args.command == "clean":
        if os.path.exists("benchmark"):
            shutil.rmtree("benchmark")
            print("Benchmark directory cleaned.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()