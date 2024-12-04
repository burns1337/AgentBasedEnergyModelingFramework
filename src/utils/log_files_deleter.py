import os
import time


# delete log files older than 7 days in the directory 'output' with the names agent_log.csv and meet_log.csv
def delete_log_files():
    current_dir = os.getcwd()
    log_dir = os.path.join(current_dir, 'output')
    current_time = time.time()
    time_period = 7 * 24 * 60 * 60  # 7 days in seconds
    log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith('agent_log') or f.startswith('meet_log')]
    for log_file in log_files:
        mtime = os.path.getmtime(log_file)
        if current_time - mtime > time_period:
            os.remove(log_file)
            print(f"Deleted log file: {log_file}")
