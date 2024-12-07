import subprocess
import pandas as pd
import re

file_path = #drive link to the path

def get_netflow_data(file_path):
    command = f"nfdump -r {file_path} -o csv"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    data = result.stdout.splitlines()
    headers = data[0].split(';')
    rows = [row.split(';') for row in data[1:]]
    df = pd.DataFrame(rows, columns=headers)
    df['bytes'] = pd.to_numeric(df['bytes'], errors='coerce')
    df['packets'] = pd.to_numeric(df['packets'], errors='coerce')
    
    return df
netflow_df = get_netflow_data('/var/log/netflow/nfcapd.202411161200')
print(netflow_df.head())

def parse_syslog(file_path):
    pattern = re.compile(r'(\w{3}\s+\d+\s\d+:\d+:\d+) (\S+) (\S+): (.*)')
    logs = []
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                timestamp, hostname, process, message = match.groups()
                logs.append({
                    'timestamp': timestamp,
                    'hostname': hostname,
                    'process': process,
                    'message': message
                })
    df = pd.DataFrame(logs)
    return df
syslog_df = parse_syslog('/var/log/syslog')
print(syslog_df.head())

#Anomaly detection using scikit (should work mostly :))
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

features = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'bytes', 'packets']
df = merged_df[features].copy()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
model = IsolationForest(contamination=0.05, random_state=42)
merged_df['anomaly'] = model.predict(df_scaled)
new_devices = merged_df[merged_df['anomaly'] == -1]
print("Detected Anomalies:")
print(new_devices)
