# clear_log.py

log_file_path = 'optimization.log'

# Open the log file in write mode to clear its contents
with open(log_file_path, 'w') as log_file:
    log_file.write('')

print(f"{log_file_path} has been cleared.")
