# clear_log.py

log_file_path = ['optimization.log','SA_log.log','smart_grid.log']

# Open the log file in write mode to clear its contents
for log_file in log_file_path:
    x = input(f'Should {log_file} be wiped of the data (Y/ any other char for n)')
    if x=='y' or x=='Y':
        with open(log_file_path, 'w') as log_file:
            log_file.write('')
    else :
        print('your log file has not been wiped.')

print(f"{log_file_path} has been cleared.")
