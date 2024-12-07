import re
import logging
from datetime import datetime

logging.basicConfig(
    filename='failure_engine.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

SYSLOG_FILE_PATH = '/var/log/syslog'  # Update this path as needed

FAILURE_PATTERNS = [
    re.compile(r'ERROR', re.IGNORECASE),
    re.compile(r'CRITICAL', re.IGNORECASE),
    re.compile(r'FAILURE', re.IGNORECASE),
    re.compile(r'DOWN', re.IGNORECASE),
]

SYSLOG_REGEX = re.compile(
    r'^(?P<month>\w{3})\s+'
    r'(?P<day>\d{1,2})\s+'
    r'(?P<time>\d{2}:\d{2}:\d{2})\s+'
    r'(?P<host>[\w\.-]+)\s+'
    r'(?P<app>\w+)(?:\[(?P<pid>\d+)\])?:\s+'
    r'(?P<message>.+)$'
)

def parse_syslog_line(line):
    match = SYSLOG_REGEX.match(line)
    if match:
        return match.groupdict()
    return None

def is_failure(message):
    for pattern in FAILURE_PATTERNS:
        if pattern.search(message):
            return True
    return False

def process_syslog_file(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parsed = parse_syslog_line(line)
                if parsed:
                    message = parsed['message']
                    if is_failure(message):
                        # Log the failure
                        failure_info = {
                            'timestamp': f"{parsed['month']} {parsed['day']} {parsed['time']}",
                            'host': parsed['host'],
                            'app': parsed['app'],
                            'pid': parsed.get('pid', ''),
                            'message': message
                        }
                        logging.info(f"Failure detected: {failure_info}")
                        print(f"Failure detected: {failure_info}")
    except FileNotFoundError:
        print(f"Syslog file not found at path: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    print("Starting Failure Engine...")
    process_syslog_file(SYSLOG_FILE_PATH)
    print("Failure Engine processing completed.")

if __name__ == "__main__":
    main()
