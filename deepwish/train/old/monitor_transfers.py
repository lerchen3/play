#!/usr/bin/env python3
"""
Transfer Monitoring Script for BPE Orchestrator
Runs on the codespace to monitor file transfers from Kaggle workers
"""

import os
import time
import json
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict
import threading
import signal
import sys

# Configuration
SYNC_DIR = Path("kaggle_work/sync")
WATCH_DIRS = [SYNC_DIR, Path('.')]
LOG_FILE = "transfer_monitor.log"
CHECK_INTERVAL = 0.1  # Check every 100ms

class TransferMonitor:
    def __init__(self):
        self.running = True
        self.file_timestamps = {}  # filepath -> first_seen_time
        self.file_sizes = {}  # filepath -> size
        self.completed_transfers = []
        self.stats = defaultdict(list)
        self.test_upload_reported = False  # ensure we only report test upload once
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s.%(msecs)03d - %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(LOG_FILE, mode='w'),
                logging.StreamHandler()
            ]
        )
        
        # Ensure sync directory exists
        for d in WATCH_DIRS:
            d.mkdir(parents=True, exist_ok=True)
        
        # Snapshot existing files to avoid measuring pre-existing ones
        initial_files = []
        for d in WATCH_DIRS:
            initial_files.extend([f for f in d.glob("*") if f.is_file()])
        now = time.time()
        for f in initial_files:
            self.file_timestamps[str(f)] = now
            size, _ = self.get_file_info(f)
            if size is not None:
                self.file_sizes[str(f)] = size
        
        logging.info("🔍 Transfer Monitor Started")
        for d in WATCH_DIRS:
            logging.info(f"📁 Monitoring directory: {d.absolute()}")
        logging.info(f"⏱️  Check interval: {CHECK_INTERVAL}s")
        
    def get_file_info(self, filepath):
        """Get file size and modification time"""
        try:
            stat = filepath.stat()
            return stat.st_size, stat.st_mtime
        except (OSError, FileNotFoundError):
            return None, None
            
    def format_size(self, bytes_size):
        """Format bytes in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f}{unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f}TB"
        
    def format_speed(self, bytes_per_sec):
        """Format transfer speed"""
        if bytes_per_sec < 1024:
            return f"{bytes_per_sec:.1f} B/s"
        elif bytes_per_sec < 1024 * 1024:
            return f"{bytes_per_sec/1024:.1f} KB/s"
        else:
            return f"{bytes_per_sec/(1024*1024):.1f} MB/s"
            
    def monitor_files(self):
        """Main monitoring loop"""
        last_file_count = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Scan all files in the watch directories
                all_files = []
                for d in WATCH_DIRS:
                    all_files.extend(d.glob("*"))
                current_file_count = len(all_files)
                
                if current_file_count != last_file_count:
                    logging.info(f"📊 File count changed: {last_file_count} → {current_file_count}")
                    last_file_count = current_file_count
                
                for filepath in all_files:
                    # Skip any log files
                    if filepath.suffix == ".log":
                        continue
                    if not filepath.is_file():
                        continue
                        
                    size, mtime = self.get_file_info(filepath)
                    if size is None:
                        continue
                        
                    file_key = str(filepath)
                    
                    # First time seeing this file
                    if file_key not in self.file_timestamps:
                        self.file_timestamps[file_key] = current_time
                        self.file_sizes[file_key] = size
                        
                        # Log the new file
                        file_type = "Unknown"
                        if "finish_" in filepath.name and filepath.suffix == ".flag":
                            file_type = "Finish Flag"
                        elif "merges_" in filepath.name and filepath.suffix == ".json":
                            file_type = "Merges"
                        elif "stats_" in filepath.name and filepath.suffix == ".json":
                            file_type = "Stats"
                            
                        logging.info(f"📥 NEW FILE: {filepath.name} ({self.format_size(size)}) - {file_type}")
                        
                        # If it's a stats file, this represents a completed upload
                        if "stats_" in filepath.name:
                            self.analyze_stats_upload(filepath, size, current_time)
                        # If it's the Kaggle test upload, parse its internal timestamp
                        if filepath.name == "uploaded_from_kaggle.txt":
                            self.analyze_test_upload(filepath, size, current_time)
                            
                    # File size changed (still being written)
                    elif self.file_sizes[file_key] != size:
                        old_size = self.file_sizes[file_key]
                        self.file_sizes[file_key] = size
                        duration = current_time - self.file_timestamps[file_key]
                        speed = size / duration if duration > 0 else 0
                        
                        logging.info(f"📈 GROWING: {filepath.name} {self.format_size(old_size)} → {self.format_size(size)} ({self.format_speed(speed)})")
                        
                        # If this is the Kaggle test upload and not yet reported, analyze it now
                        if filepath.name == "uploaded_from_kaggle.txt" and not self.test_upload_reported:
                            self.analyze_test_upload(filepath, size, current_time)
                            self.test_upload_reported = True
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"❌ Monitor error: {e}")
                
            time.sleep(CHECK_INTERVAL)
            
    def analyze_stats_upload(self, filepath, size, completion_time):
        """Analyze a completed stats file upload"""
        # Extract rank and iteration from filename
        # Expected format: stats_{rank}_{iteration}.json
        name_parts = filepath.stem.split('_')
        if len(name_parts) >= 3:
            try:
                rank = int(name_parts[1])
                iteration = int(name_parts[2])
                
                # Check if we have the corresponding finish flag
                flag_file = SYNC_DIR / f"finish_{iteration}.flag"
                flag_exists = flag_file.exists()
                
                if flag_exists:
                    flag_time = self.file_timestamps.get(str(flag_file))
                    if flag_time:
                        response_time = completion_time - flag_time
                        self.stats['response_times'].append(response_time)
                        
                        logging.info(f"⚡ UPLOAD COMPLETE: Rank {rank}, Iteration {iteration}")
                        logging.info(f"   📏 Size: {self.format_size(size)}")
                        logging.info(f"   ⏱️  Response time: {response_time:.2f}s (from flag to stats)")
                        
                        if response_time > 10:
                            logging.warning(f"⚠️  SLOW UPLOAD: {response_time:.2f}s for {self.format_size(size)}")
                            
                self.stats['upload_sizes'].append(size)
                self.stats['upload_times'].append(completion_time)
                
            except (ValueError, IndexError):
                logging.warning(f"⚠️  Could not parse filename: {filepath.name}")
                
    def analyze_test_upload(self, filepath, size, detection_time):
        """Analyze the Kaggle test upload by parsing its embedded timestamp"""
        try:
            content = filepath.read_text().splitlines()
            ts_line = next((l for l in content if l.startswith("Current time:")), None)
            if not ts_line:
                return
            ts_str = ts_line.split("Current time:",1)[1].strip()
            
            # Normalize multiple spaces to single space for parsing
            normalized_ts = ' '.join(ts_str.split())
            
            # Try multiple format variations
            formats = [
                "%a %b %d %I:%M:%S %p %Z %Y",  # 12-hour with AM/PM
                "%a %b %d %H:%M:%S %Z %Y",     # 24-hour format
                "%a %b %d %H:%M:%S %p %Z %Y"   # Mixed format (shouldn't happen but just in case)
            ]
            
            created_ts = None
            for fmt in formats:
                try:
                    dt = datetime.strptime(normalized_ts, fmt)
                    created_ts = dt.timestamp()
                    break
                except Exception as e:
                    continue
            
            if created_ts is None:
                try:
                    created_ts = float(normalized_ts)
                except Exception:
                    logging.warning(f"⚠️ Could not parse timestamp: {ts_str} (normalized: {normalized_ts})")
                    return
            
            latency = detection_time - created_ts
            rate = size / latency if latency > 0 else 0
            logging.info(f"🔄 TEST UPLOAD: {filepath.name}")
            logging.info(f"   📏 Size: {self.format_size(size)}")
            logging.info(f"   🕑 Created: {ts_str}")
            logging.info(f"   ⏲️  Latency: {latency:.2f}s")
            logging.info(f"   🏎️  Rate: {self.format_speed(rate)}")
        except Exception as e:
            logging.error(f"❌ Error analyzing test upload: {e}")
        
    def print_stats(self):
        """Print summary statistics"""
        if not self.stats['response_times']:
            logging.info("📊 No completed transfers yet")
            return
            
        response_times = self.stats['response_times']
        upload_sizes = self.stats['upload_sizes']
        
        logging.info(f"\n📊 TRANSFER STATISTICS:")
        logging.info(f"   Completed uploads: {len(response_times)}")
        logging.info(f"   Average response time: {sum(response_times)/len(response_times):.2f}s")
        logging.info(f"   Min response time: {min(response_times):.2f}s")
        logging.info(f"   Max response time: {max(response_times):.2f}s")
        if upload_sizes:
            logging.info(f"   Average file size: {self.format_size(sum(upload_sizes)/len(upload_sizes))}")
            logging.info(f"   Total data transferred: {self.format_size(sum(upload_sizes))}")
            
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        logging.info("\n🛑 Shutting down monitor...")
        self.running = False
        self.print_stats()
        sys.exit(0)
        
def main():
    monitor = TransferMonitor()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, monitor.signal_handler)
    signal.signal(signal.SIGTERM, monitor.signal_handler)
    
    # Start monitoring
    try:
        monitor.monitor_files()
    except KeyboardInterrupt:
        monitor.signal_handler(None, None)

if __name__ == "__main__":
    main() 