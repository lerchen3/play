# Distributed BPE orchestrator running on GitHub Codespace

import os
import json
import subprocess
subprocess.run("pip install kaggle", shell=True)
import shutil
import time
import numpy as np
from collections import Counter
from pathlib import Path
import logging
from datetime import datetime
import sys
from datasets import load_dataset

# ==== CONFIGURATION ====
# Default configuration (can be overridden by config.py)
GITHUB_TOKEN = ""
with open("accounts.json", "r") as f:
    KAGGLE_ACCOUNTS = json.load(f)
VOCAB_SIZE = 50000 #~sqrt(corpus size in tokens, which is 0.75e9)
INITIAL_VOCAB_SIZE = 256
KERNEL_NAME_BASE = "bpeworkerworking"
NUM_RANKS_PER_ACCOUNT = 5 # 5 cpu instances per account (capped)
STATS_TIMEOUT = 600
KERNEL_STARTUP_DELAY = 30

# ==== DERIVED CONFIGURATION ====
WORK_DIR = Path(__file__).parent / "kaggle_work"
SYNC_DIR = WORK_DIR / "sync"
RESULTS_DIR = WORK_DIR / "results"

# Codespace info (will be auto-detected)
CODESPACE_NAME = "reimagined-goggles-wjwxwx7jx74h5xqx"
CODESPACE_REPO = "arch"

# ==== TIMING SETUP ====
def setup_logging():
    """Setup logging with timestamps"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bpe_orchestrator.log'),
            logging.StreamHandler()
        ]
    )

def log_timing(operation_name, start_time, end_time=None):
    """Log timing information for operations"""
    if end_time is None:
        end_time = time.time()
    duration = end_time - start_time
    logging.info(f"⏱️  {operation_name}: {duration:.2f} seconds")
    return duration

def log_start(operation_name):
    """Log start of operation and return start time"""
    start_time = time.time()
    logging.info(f"🚀 Starting: {operation_name}")
    return start_time

# =======================

def run(cmd, **kwargs):
    """Execute command and print it"""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, **kwargs)
    return result

def setup_github_cli():
    """Setup GitHub CLI authentication"""
    start_time = log_start("GitHub CLI setup")
    
    if not GITHUB_TOKEN:
        print("GitHub token not set. Skipping gh auth.")
        return
    proc = subprocess.Popen(
        "gh auth login --with-token", shell=True, stdin=subprocess.PIPE, text=True
    )
    proc.communicate(GITHUB_TOKEN)
    proc.wait()
    
    log_timing("GitHub CLI setup", start_time)

def setup_kaggle_creds(creds):
    """Setup Kaggle credentials"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    cred_path = kaggle_dir / "kaggle.json"
    with open(cred_path, "w") as f:
        json.dump(creds, f)
    os.chmod(cred_path, 0o600)

def create_kernel_notebook(rank, total_ranks):
    """Create rank-specific kernel notebook with C++ BPE implementation"""
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Install GitHub CLI and setup authentication\n",
                    "import os, subprocess, time, json\n",
                    "import numpy as np\n",
                    "from collections import Counter\n",
                    "from datasets import load_dataset\n",
                    "import multiprocessing\n",
                    "\n",
                    "print('Installing GitHub CLI...')\n",
                    "# Download and install GitHub CLI\n",
                    "!wget https://github.com/cli/cli/releases/download/v2.74.0/gh_2.74.0_linux_amd64.tar.gz -O ghcli.tar.gz\n",
                    "!tar -xf ghcli.tar.gz\n",
                    "!mkdir -p ~/.local/bin\n",
                    "!cp gh_2.74.0_linux_amd64/bin/gh ~/.local/bin/gh\n",
                    "!rm -rf ghcli.tar.gz gh_2.74.0_linux_amd64\n",
                    "\n",
                    "# Setup GitHub token authentication\n",
                    f"os.environ['GH_TOKEN'] = '{GITHUB_TOKEN}'\n",
                    "\n",
                    "# Verify installation\n",
                    "!~/.local/bin/gh --version\n",
                    "!~/.local/bin/gh auth status\n",
                    "\n",
                    f"RANK = {rank}\n",
                    f"TOTAL_RANKS = {total_ranks}\n",
                    f"CODESPACE_NAME = '{CODESPACE_NAME}'\n",
                    f"CODESPACE_REPO = '{CODESPACE_REPO}'\n",
                    "NUM_THREADS = multiprocessing.cpu_count()\n",
                    "\n",
                    "print(f'Worker rank {RANK}/{TOTAL_RANKS} starting with {NUM_THREADS} CPU threads...')"
                ]
            },
            {
                "cell_type": "code", 
                "metadata": {},
                "source": [
                    "# Load dataset portion directly from HuggingFace\n",
                    "import time\n",
                    "from datasets import load_dataset\n",
                    "start_time = time.time()\n",
                    "dataset = load_dataset('open-r1/OpenR1-Math-220k', split='train')\n",
                    "dataset_load_time = time.time() - start_time\n",
                    "num_examples = len(dataset)\n",
                    "chunk_size = (num_examples + TOTAL_RANKS - 1) // TOTAL_RANKS\n",
                    "start_idx = RANK * chunk_size\n",
                    "end_idx = min(start_idx + chunk_size, num_examples)\n",
                    "slice_start_time = time.time()\n",
                    "corpus_data = []\n",
                    "for i in range(start_idx, end_idx):\n",
                    "    corpus_data.extend(dataset[i]['generations'])\n",
                    "slice_time = time.time() - slice_start_time\n",
                    "print(f'Rank {RANK}/{TOTAL_RANKS} processing items {start_idx}:{end_idx} ({len(corpus_data)} items)')\n",
                    "print(f'Dataset download: {dataset_load_time:.2f}s')\n",
                    "print(f'Data slicing: {slice_time:.2f}s')\n",
                    "\n",
                    "# Save corpus to text file for initial C++ loading\n",
                    "save_start = time.time()\n",
                    "with open('corpus.txt', 'w', encoding='utf-8') as f:\n",
                    "    for text in corpus_data:\n",
                    "        f.write(text.replace('\\n', '\\\\n') + '\\n')\n",
                    "save_time = time.time() - save_start\n",
                    "total_time = time.time() - start_time\n",
                    "print(f'Saved corpus to corpus.txt in {save_time:.2f}s')\n",
                    "print(f'Total data preparation time: {total_time:.2f}s')\n",
                    "\n",
                    "# Clear memory\n",
                    "del dataset, corpus_data\n",
                    "print('Cleared dataset from memory')\n"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Create optimized C++ BPE worker that keeps token arrays in memory\n",
                    "cpp_code = '''\n",
                    "#include <bits/stdc++.h>\n",
                    "#include <thread>\n",
                    "#include <mutex>\n",
                    "#include <regex>\n",
                    "using namespace std;\n",
                    "\n",
                    "class IncrementalBPEWorker {\n",
                    "private:\n",
                    "    vector<vector<int>> docs;\n",
                    "    int num_threads;\n",
                    "    int rank;\n",
                    "    \n",
                    "public:\n",
                    "    IncrementalBPEWorker(const string& corpus_file, int threads, int worker_rank) \n",
                    "        : num_threads(threads), rank(worker_rank) {\n",
                    "        loadCorpus(corpus_file);\n",
                    "    }\n",
                    "    \n",
                    "    void loadCorpus(const string& corpus_file) {\n",
                    "        auto start = chrono::high_resolution_clock::now();\n",
                    "        \n",
                    "        ifstream fin(corpus_file);\n",
                    "        if (!fin.is_open()) {\n",
                    "            cerr << \"Error: Cannot open corpus file: \" << corpus_file << endl;\n",
                    "            exit(1);\n",
                    "        }\n",
                    "        \n",
                    "        string line;\n",
                    "        while(getline(fin, line)) {\n",
                    "            vector<int> doc;\n",
                    "            doc.reserve(line.size());\n",
                    "            for(char c : line) {\n",
                    "                doc.push_back((unsigned char)c);\n",
                    "            }\n",
                    "            docs.push_back(move(doc));\n",
                    "        }\n",
                    "        fin.close();\n",
                    "        \n",
                    "        auto end = chrono::high_resolution_clock::now();\n",
                    "        double load_time = chrono::duration<double>(end - start).count();\n",
                    "        cout << \"Loaded \" << docs.size() << \" documents in \" << load_time << \"s\" << endl;\n",
                    "    }\n",
                    "    \n",
                    "    void applyMerges(const vector<pair<pair<int,int>, int>>& merges) {\n",
                    "        if(merges.empty()) return;\n",
                    "        \n",
                    "        auto start = chrono::high_resolution_clock::now();\n",
                    "        \n",
                    "        // Build merge map for O(1) lookup\n",
                    "        unordered_map<long long, int> merge_map;\n",
                    "        for(const auto& merge : merges) {\n",
                    "            long long key = ((long long)merge.first.first << 32) | (unsigned int)merge.first.second;\n",
                    "            merge_map[key] = merge.second;\n",
                    "        }\n",
                    "        \n",
                    "        // Apply merges in parallel\n",
                    "        int N = docs.size();\n",
                    "        auto merge_worker = [&](int tid) {\n",
                    "            int start_idx = tid * N / num_threads;\n",
                    "            int end_idx = (tid + 1) * N / num_threads;\n",
                    "            \n",
                    "            for(int i = start_idx; i < end_idx; ++i) {\n",
                    "                auto& tokens = docs[i];\n",
                    "                vector<int> new_tokens;\n",
                    "                new_tokens.reserve(tokens.size());\n",
                    "                \n",
                    "                for(size_t j = 0; j < tokens.size(); ) {\n",
                    "                    if(j + 1 < tokens.size()) {\n",
                    "                        long long key = ((long long)tokens[j] << 32) | (unsigned int)tokens[j+1];\n",
                    "                        auto it = merge_map.find(key);\n",
                    "                        if(it != merge_map.end()) {\n",
                    "                            new_tokens.push_back(it->second);\n",
                    "                            j += 2;\n",
                    "                            continue;\n",
                    "                        }\n",
                    "                    }\n",
                    "                    new_tokens.push_back(tokens[j]);\n",
                    "                    j++;\n",
                    "                }\n",
                    "                tokens.swap(new_tokens);\n",
                    "            }\n",
                    "        };\n",
                    "        \n",
                    "        vector<thread> threads;\n",
                    "        for(int t = 0; t < num_threads; ++t) {\n",
                    "            threads.emplace_back(merge_worker, t);\n",
                    "        }\n",
                    "        for(auto& th : threads) {\n",
                    "            th.join();\n",
                    "        }\n",
                    "        \n",
                    "        auto end = chrono::high_resolution_clock::now();\n",
                    "        double merge_time = chrono::duration<double>(end - start).count();\n",
                    "        cout << \"Applied \" << merges.size() << \" merges in \" << merge_time << \"s\" << endl;\n",
                    "    }\n",
                    "    \n",
                    "    void countPairsAndSave(const string& output_file, int top_k = 1000) {\n",
                    "        auto start = chrono::high_resolution_clock::now();\n",
                    "        \n",
                    "        unordered_map<long long, long long> global_counts;\n",
                    "        mutex mtx;\n",
                    "        \n",
                    "        int N = docs.size();\n",
                    "        auto count_worker = [&](int tid) {\n",
                    "            int start_idx = tid * N / num_threads;\n",
                    "            int end_idx = (tid + 1) * N / num_threads;\n",
                    "            unordered_map<long long, long long> local_counts;\n",
                    "            \n",
                    "            for(int i = start_idx; i < end_idx; ++i) {\n",
                    "                const auto& tokens = docs[i];\n",
                    "                for(size_t j = 0; j + 1 < tokens.size(); ++j) {\n",
                    "                    long long key = ((long long)tokens[j] << 32) | (unsigned int)tokens[j+1];\n",
                    "                    local_counts[key]++;\n",
                    "                }\n",
                    "            }\n",
                    "            \n",
                    "            lock_guard<mutex> lock(mtx);\n",
                    "            for(const auto& kv : local_counts) {\n",
                    "                global_counts[kv.first] += kv.second;\n",
                    "            }\n",
                    "        };\n",
                    "        \n",
                    "        vector<thread> threads;\n",
                    "        for(int t = 0; t < num_threads; ++t) {\n",
                    "            threads.emplace_back(count_worker, t);\n",
                    "        }\n",
                    "        for(auto& th : threads) {\n",
                    "            th.join();\n",
                    "        }\n",
                    "        \n",
                    "        // Get top-k pairs\n",
                    "        vector<pair<long long, long long>> vec(global_counts.begin(), global_counts.end());\n",
                    "        sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {\n",
                    "            return a.second > b.second;\n",
                    "        });\n",
                    "        \n",
                    "        int K = min((int)vec.size(), top_k);\n",
                    "        \n",
                    "        // Save as JSON (matches Python format: {\\\"65,66\\\": count})\n",
                    "        ofstream fout(output_file);\n",
                    "        if (!fout.is_open()) {\n",
                    "            cerr << \"Error: Cannot write to output file: \" << output_file << endl;\n",
                    "            return;\n",
                    "        }\n",
                    "        \n",
                    "        fout << '{';\n",
                    "        for(int i = 0; i < K; ++i) {\n",
                    "            if(i > 0) fout << \",\";\n",
                    "            int first = (int)(vec[i].first >> 32);\n",
                    "            int second = (int)(vec[i].first & 0xFFFFFFFF);\n",
                    "            fout << '\"' << first << ',' << second << '\"' << ':' << vec[i].second;\n",
                    "        }\n",
                    "        fout << '}';\n",
                    "        fout.close();\n",
                    "        \n",
                    "        auto end = chrono::high_resolution_clock::now();\n",
                    "        double count_time = chrono::duration<double>(end - start).count();\n",
                    "        cout << \"Counted \" << global_counts.size() << \" unique pairs, saved top-\" << K << \" in \" << count_time << \"s\" << endl;\n",
                    "    }\n",
                    "    \n",
                    "    // Fixed JSON parser using regex for robustness\n",
                    "    vector<pair<pair<int,int>, int>> parseJsonMerges(const string& json_str) {\n",
                    "        vector<pair<pair<int,int>, int>> merges;\n",
                    "        \n",
                    "        if (json_str == \\\"[]\\\" || json_str.empty()) {\n",
                    "            return merges;\n",
                    "        }\n",
                    "        \n",
                    "        // Use regex to parse JSON: [{\\\"pair\\\":[65,66],\\\"id\\\":256}, ...]\n",
                    "        regex merge_regex(R\\\"(\\{\\\"pair\\\":\\[(\\d+),(\\d+)\\],\\\"id\\\":(\\d+)\\})\\\");\n",
                    "        sregex_iterator iter(json_str.begin(), json_str.end(), merge_regex);\n",
                    "        sregex_iterator end;\n",
                    "        \n",
                    "        for (; iter != end; ++iter) {\n",
                    "            const smatch& match = *iter;\n",
                    "            try {\n",
                    "                int first = stoi(match[1].str());\n",
                    "                int second = stoi(match[2].str());\n",
                    "                int id = stoi(match[3].str());\n",
                    "                merges.push_back({{first, second}, id});\n",
                    "            } catch (const exception& e) {\n",
                    "                cerr << \\\"Error parsing merge: \\\" << match.str() << \\\" - \\\" << e.what() << endl;\n",
                    "            }\n",
                    "        }\n",
                    "        \n",
                    "        return merges;\n",
                    "    }\n",
                    "};\n",
                    "\n",
                    "int main(int argc, char** argv) {\n",
                    "    if(argc < 4) {\n",
                    "        cerr << \\\"Usage: ./bpe_worker corpus.txt num_threads rank\\\" << endl;\n",
                    "        return 1;\n",
                    "    }\n",
                    "    \n",
                    "    string corpus_file = argv[1];\n",
                    "    int num_threads = stoi(argv[2]);\n",
                    "    int rank = stoi(argv[3]);\n",
                    "    \n",
                    "    cout << \\\"Starting BPE worker: rank=\\\" << rank << \\\", threads=\\\" << num_threads << endl;\n",
                    "    \n",
                    "    IncrementalBPEWorker worker(corpus_file, num_threads, rank);\n",
                    "    \n",
                    "    // Initial stats (iteration 0)\n",
                    "    worker.countPairsAndSave(\\\"stats_\\\" + to_string(rank) + \\\"_0.json\\\");\n",
                    "    cout << \\\"Generated initial stats for rank \\\" << rank << endl;\n",
                    "    \n",
                    "    // Main worker loop - wait for merge instructions\n",
                    "    int iteration = 1;\n",
                    "    while(true) {\n",
                    "        string merge_file = \\\"merges_\\\" + to_string(iteration) + \\\".json\\\";\n",
                    "        \n",
                    "        // Poll for merge file with exponential backoff\n",
                    "        int wait_ms = 100;\n",
                    "        while(true) {\n",
                    "            ifstream check(merge_file);\n",
                    "            if(check.good()) {\n",
                    "                check.close();\n",
                    "                break;\n",
                    "            }\n",
                    "            this_thread::sleep_for(chrono::milliseconds(wait_ms));\n",
                    "            wait_ms = min(wait_ms * 2, 1000); // Cap at 1 second\n",
                    "        }\n",
                    "        \n",
                    "        // Read merge instructions\n",
                    "        ifstream fin(merge_file);\n",
                    "        string json_content((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());\n",
                    "        fin.close();\n",
                    "        \n",
                    "        if(json_content == \\\"[]\\\" || json_content.empty()) {\n",
                    "            cout << \\\"No more merges, stopping at iteration \\\" << iteration << endl;\n",
                    "            break;\n",
                    "        }\n",
                    "        \n",
                    "        // Parse merges using robust regex parser\n",
                    "        auto merges = worker.parseJsonMerges(json_content);\n",
                    "        \n",
                    "        cout << \\\"Iteration \\\" << iteration << \\\": applying \\\" << merges.size() << \\\" merges\\\" << endl;\n",
                    "        \n",
                    "        if (merges.empty()) {\n",
                    "            cout << \\\"Warning: No valid merges parsed from JSON\\\" << endl;\n",
                    "        }\n",
                    "        \n",
                    "        // Apply merges\n",
                    "        worker.applyMerges(merges);\n",
                    "        \n",
                    "        // Count new pairs and save stats\n",
                    "        string stats_file = \\\"stats_\\\" + to_string(rank) + \\\"_\\\" + to_string(iteration) + \\\".json\\\";\n",
                    "        worker.countPairsAndSave(stats_file);\n",
                    "        \n",
                    "        iteration++;\n",
                    "    }\n",
                    "    \n",
                    "    cout << \\\"BPE worker rank \\\" << rank << \\\" completed successfully\\\" << endl;\n",
                    "    return 0;\n",
                    "}\n",
                    "'''\n",
                    "\n",
                    "# Write C++ code to file\n",
                    "with open('bpe_worker.cpp', 'w') as f:\n",
                    "    f.write(cpp_code)\n",
                    "\n",
                    "print('Created optimized C++ BPE worker code')\n"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Compile C++ worker\n",
                    "import subprocess\n",
                    "\n",
                    "print('Compiling C++ BPE worker...')\n",
                    "result = subprocess.run(['g++', '-O3', '-march=native', '-std=c++17', '-pthread', \n",
                    "                        'bpe_worker.cpp', '-o', 'bpe_worker'], \n",
                    "                       capture_output=True, text=True)\n",
                    "if result.returncode != 0:\n",
                    "    print('Compilation failed:')\n",
                    "    print(result.stderr)\n",
                    "    raise Exception('C++ compilation failed')\n",
                    "else:\n",
                    "    print('C++ compilation successful!')\n"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Main BPE worker loop with C++ backend and sync coordination\n",
                    "import time, json, subprocess, os\n",
                    "\n",
                    "iteration = 0\n",
                    "print(f'Starting C++ BPE worker loop for rank {RANK}...')\n",
                    "\n",
                    "# Start C++ worker process in background\n",
                    "cpp_process = subprocess.Popen(['./bpe_worker', 'corpus.txt', str(NUM_THREADS), str(RANK)], \n",
                    "                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n",
                    "\n",
                    "print('C++ worker started, entering sync loop...')\n",
                    "\n",
                    "while True:\n",
                    "    iteration_start = time.time()\n",
                    "    \n",
                    "    # Wait for finish flag from orchestrator by copying it locally with a timeout\n",
                    "    flag_remote = f\"remote:workspaces/{CODESPACE_REPO}/kaggle_work/sync/finish_{iteration}.flag\"\n",
                    "    flag_local = f\"finish_{iteration}.flag\"\n",
                    "    print(f'Debug: CODESPACE_NAME={CODESPACE_NAME}, CODESPACE_REPO={CODESPACE_REPO}, iteration={iteration}')\n",
                    "    print(f'Debug: flag_remote={flag_remote}, flag_local={flag_local}')\n",
                    "    cmd = f\"~/.local/bin/gh codespace ssh -c {CODESPACE_NAME} -- \\\"cat /workspaces/{CODESPACE_REPO}/kaggle_work/sync/finish_{iteration}.flag\\\" > {flag_local}\"\n",
                    "    print(f'Debug: full command = {cmd}')\n",
                    "    while True:\n",
                    "        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n",
                    "        print(f'Debug: gh ssh command returned {res.returncode}')\n",
                    "        if res.stderr: print(f'Debug: stderr = {res.stderr}')\n",
                    "        print(f'Debug: checking for local file {flag_local}, exists = {os.path.exists(flag_local)}')\n",
                    "        # If copy succeeded and the file exists locally, proceed\n",
                    "        if res.returncode == 0 and os.path.exists(flag_local):\n",
                    "            break\n",
                    "        time.sleep(1)\n",
                    "        if time.time() - iteration_start > 600:\n",
                    "            print('Timeout waiting for finish flag, stopping...')\n",
                    "            raise Exception('Timeout waiting for finish flag')\n",
                    "            break\n",
                    "    \n",
                    "    print(f'Iteration {iteration}: finish flag detected')\n",
                    "    \n",
                    "    # Fetch merges for this iteration by copying remote file locally\n",
                    "    merge_remote = f\"remote:workspaces/{CODESPACE_REPO}/kaggle_work/sync/merges_{iteration}.json\"\n",
                    "    merge_local = f\"merges_{iteration}.json\"\n",
                    "    subprocess.run(f\"~/.local/bin/gh codespace ssh -c {CODESPACE_NAME} -- \\\"cat /workspaces/{CODESPACE_REPO}/kaggle_work/sync/merges_{iteration}.json\\\" > {merge_local}\", shell=True)\n",
                    "    # Read merges JSON from local file\n",
                    "    if os.path.exists(merge_local):\n",
                    "        with open(merge_local) as f:\n",
                    "            merges_json = f.read().strip() or '[]'\n",
                    "    else:\n",
                    "        raise Exception(f'Merge file {merge_local} not found')\n",
                    "    \n",
                    "    # Write merges file for C++ worker\n",
                    "    with open(f'merges_{iteration}.json', 'w') as f:\n",
                    "        f.write(merges_json)\n",
                    "    \n",
                    "    merges = json.loads(merges_json)\n",
                    "    if iteration > 0 and not merges:\n",
                    "        print(f'No merges for iteration {iteration}, stopping.')\n",
                    "        cpp_process.terminate()\n",
                    "        break\n",
                    "    \n",
                    "    print(f'Iteration {iteration}: wrote {len(merges)} merges for C++ worker')\n",
                    "    \n",
                    "    # Wait for C++ worker to generate stats file\n",
                    "    stats_file = f'stats_{RANK}_{iteration}.json'\n",
                    "    while not os.path.exists(stats_file):\n",
                    "        time.sleep(0.1)\n",
                    "    \n",
                    "    print(f'Iteration {iteration}: C++ worker completed, uploading stats')\n",
                    "    \n",
                    "    # Upload stats to orchestrator\n",
                    "    upload_start = time.time()\n",
                    "    cmd = f\"~/.local/bin/gh codespace ssh -c {CODESPACE_NAME} -- \\\"cat > /workspaces/{CODESPACE_REPO}/kaggle_work/sync/stats_{RANK}_{iteration}.json\\\" < {stats_file}\"\n",
                    "    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n",
                    "    if res.returncode != 0:\n",
                    "        print(f'Stats upload failed: {res.stderr}')\n",
                    "    else:\n",
                    "        upload_time = time.time() - upload_start\n",
                    "        iteration_total = time.time() - iteration_start\n",
                    "        print(f'Rank {RANK} iteration {iteration} complete (upload: {upload_time:.2f}s, total: {iteration_total:.2f}s)')\n",
                    "    \n",
                    "    # Clean up local files\n",
                    "    os.remove(stats_file)\n",
                    "    os.remove(f'merges_{iteration}.json')\n",
                    "    \n",
                    "    iteration += 1\n",
                    "\n",
                    "# Get any remaining output from C++ process\n",
                    "if cpp_process.poll() is None:\n",
                    "    cpp_process.terminate()\n",
                    "    \n",
                    "stdout, _ = cpp_process.communicate()\n",
                    "print(\"C++ worker output:\")\n",
                    "print(stdout)\n",
                    "\n",
                    "print(f'Rank {RANK} BPE worker completed!')\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "python3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.x"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    return notebook

def prepare_kernel_for_rank(account, rank, total_ranks):
    """Prepare kernel directory and metadata for specific rank"""
    start_time = log_start(f"Kernel preparation for rank {rank}")
    
    slug = f"{account['username']}/{KERNEL_NAME_BASE}_{rank}"
    kdir = WORK_DIR / f"{KERNEL_NAME_BASE}_{rank}"
    kdir.mkdir(parents=True, exist_ok=True)
    
    # Create rank-specific notebook
    notebook = create_kernel_notebook(rank, total_ranks)
    with open(kdir / "kaggle_kernel.ipynb", "w") as f:
        json.dump(notebook, f, indent=2)
    
    # Create metadata
    meta = {
        "id": slug,
        "title": f"{KERNEL_NAME_BASE}_{rank}",
        "code_file": "kaggle_kernel.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
    }
    with open(kdir / "kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    log_timing(f"Kernel preparation for rank {rank}", start_time)
    return kdir, slug

def push_and_start_kernel(kdir, account):
    """Push and start a kernel"""
    start_time = log_start(f"Kernel push for {kdir.name}")
    
    # Set credentials for this specific account before pushing
    setup_kaggle_creds(account)
    
    run(f"kaggle kernels push -p {kdir}")
    time.sleep(2)  # Brief pause between push and start
    
    log_timing(f"Kernel push for {kdir.name}", start_time)

def get_total_ranks():
    """Calculate total number of ranks across all accounts"""
    return len(KAGGLE_ACCOUNTS) * NUM_RANKS_PER_ACCOUNT

def setup_sync_directory():
    """Setup synchronization directory"""
    start_time = log_start("Sync directory setup")
    
    if SYNC_DIR.exists():
        shutil.rmtree(SYNC_DIR)
    SYNC_DIR.mkdir(parents=True)
    
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True)
    
    log_timing("Sync directory setup", start_time)

def wait_for_all_stats(iteration, total_ranks, timeout=None):
    """Wait for all ranks to submit their statistics"""
    if timeout is None:
        timeout = STATS_TIMEOUT
    
    start_time = log_start(f"Waiting for stats from {total_ranks} ranks (iteration {iteration})")
    
    start_time_actual = time.time()
    while time.time() - start_time_actual < timeout:
        stats_files = list(SYNC_DIR.glob(f"stats_*_{iteration}.json"))
        if len(stats_files) == total_ranks:
            duration = log_timing(f"All {total_ranks} ranks completed iteration {iteration}", start_time)
            logging.info(f"📊 Rank completion stats: {len(stats_files)}/{total_ranks} in {duration:.2f}s")
            return True
        
        logging.info(f"Have {len(stats_files)}/{total_ranks} stats files...")
        time.sleep(5)
    
    duration = time.time() - start_time_actual
    logging.warning(f"⚠️  Timeout waiting for stats (got {len(stats_files)}/{total_ranks} in {duration:.2f}s)")
    return False

def aggregate_stats(iteration, total_ranks):
    """Aggregate statistics from all ranks"""
    start_time = log_start(f"Aggregating stats for iteration {iteration}")
    
    total_counts = Counter()
    files_processed = 0
    
    for rank in range(total_ranks):
        stats_file = SYNC_DIR / f"stats_{rank}_{iteration}.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            
            # Convert back to tuple keys and aggregate
            for pair_str, count in stats.items():
                if ',' in pair_str:
                    pair = tuple(map(int, pair_str.split(',')))
                    total_counts[pair] += count
            files_processed += 1
    
    duration = log_timing(f"Stats aggregation ({files_processed} files, {len(total_counts)} unique pairs)", start_time)
    logging.info(f"📈 Aggregated {sum(total_counts.values())} total pair occurrences")
    
    return total_counts

def run_distributed_bpe():
    """Main distributed BPE algorithm with incremental merges"""
    start_time = log_start("Distributed BPE algorithm")
    total_ranks = get_total_ranks()
    vocab = {}  # (token1, token2) -> new_token_id
    current_vocab_size = INITIAL_VOCAB_SIZE
    iteration = 0

    # Publish initial empty merge batch and signal iteration 0
    merges_file = SYNC_DIR / f"merges_{iteration}.json"
    with open(merges_file, 'w') as f:
        json.dump([], f)
    (SYNC_DIR / f"finish_{iteration}.flag").touch()

    iteration_times = []
    while current_vocab_size < VOCAB_SIZE:
        # Wait for stats from workers
        if not wait_for_all_stats(iteration, total_ranks):
            logging.error("Failed to receive all stats, stopping...")
            break

        # Aggregate statistics for this iteration
        total_counts = aggregate_stats(iteration, total_ranks)
        if not total_counts:
            logging.warning("No pairs found, stopping BPE")
            break

        selected = []
        used_first = set()
        used_second = set()
        for pair, _ in total_counts.most_common():
            if pair[0] not in used_second and pair[1] not in used_first:
                selected.append(pair)
                used_first.add(pair[0])
                used_second.add(pair[1])
        if not selected:
            logging.warning("No valid merges selected, stopping")
            break

        # Assign new IDs and update vocab
        for pair in selected:
            vocab[pair] = current_vocab_size # tokens are 0 indexed; this is the new token id.
            current_vocab_size += 1

        # Prepare for next iteration
        iteration += 1
        iteration_times.append(time.time() - start_time)

        # Publish new merge batch and signal next iteration
        merges_list = [{"pair": [p[0], p[1]], "id": vocab[p]} for p in selected]
        merges_file = SYNC_DIR / f"merges_{iteration}.json"
        with open(merges_file, 'w') as f:
            json.dump(merges_list, f)
        (SYNC_DIR / f"finish_{iteration}.flag").touch()

    # Save final vocabulary
    save_start = time.time()
    final_vocab_file = RESULTS_DIR / "final_vocab.json"
    with open(final_vocab_file, 'w') as f:
        vocab_serializable = {f'{k[0]},{k[1]}': v for k, v in vocab.items()}
        json.dump(vocab_serializable, f, indent=2)
    save_time = time.time() - save_start

    total_time = log_timing("Complete distributed BPE algorithm", start_time)
    
    # Log summary statistics
    if iteration_times:
        avg_iteration = sum(iteration_times) / len(iteration_times)
        min_iteration = min(iteration_times)
        max_iteration = max(iteration_times)
        logging.info(f"\n📊 BPE TIMING SUMMARY:")
        logging.info(f"   Total runtime: {total_time:.2f}s")
        logging.info(f"   Iterations completed: {len(iteration_times)}")
        logging.info(f"   Average iteration time: {avg_iteration:.2f}s")
        logging.info(f"   Fastest iteration: {min_iteration:.2f}s")
        logging.info(f"   Slowest iteration: {max_iteration:.2f}s")
        logging.info(f"   Final vocab save: {save_time:.2f}s")
    
    logging.info(f"\nBPE complete! Final vocabulary saved to {final_vocab_file}")

def main():
    """Main orchestration function"""
    setup_logging()
    overall_start = time.time()
    
    if not KAGGLE_ACCOUNTS:
        logging.error("ERROR: KAGGLE_ACCOUNTS is empty. Please configure your Kaggle credentials.")
        return
    
    logging.info("Setting up distributed BPE orchestrator...")
    logging.info("Note: Each kernel will fetch its own data portion directly from HuggingFace")
    
    # Setup
    setup_github_cli()
    setup_sync_directory()
    
    total_ranks = get_total_ranks()
    
    # Setup kernels for each account
    kernel_setup_start = log_start("Kernel setup phase")
    kernels_to_start = []
    rank = 0
    
    for account in KAGGLE_ACCOUNTS:
        for local_rank in range(NUM_RANKS_PER_ACCOUNT):
            kdir, slug = prepare_kernel_for_rank(account, rank, total_ranks)
            kernels_to_start.append((kdir, slug, account))  # Include account info
            rank += 1
    
    # Push and start all kernels
    push_start = log_start(f"Pushing {len(kernels_to_start)} kernels")
    for kdir, slug, account in kernels_to_start:
        push_and_start_kernel(kdir, account)  # Pass account to set creds
        logging.info(f"Started kernel: {slug}")
    log_timing(f"All kernel pushes", push_start)
    
    log_timing("Complete kernel setup phase", kernel_setup_start)
    
    # Give kernels time to start up and download data
    startup_start = log_start("Kernel startup delay")
    time.sleep(KERNEL_STARTUP_DELAY)
    log_timing(f"Kernel startup delay ({KERNEL_STARTUP_DELAY}s)", startup_start)
    
    # Run distributed BPE
    run_distributed_bpe()
    
    total_runtime = log_timing("COMPLETE ORCHESTRATION", overall_start)
    logging.info(f"🎉 Distributed BPE orchestration completed in {total_runtime:.2f}s")

if __name__ == "__main__":
    main()
