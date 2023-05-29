import pandas as pd
import re
import matplotlib.pyplot as plt
from tabulate import tabulate

log_file = "logfile.txt"
with open(log_file, "r") as file:
    log_data = file.read()

lines = log_data.split("\n")

pattern = r"\[([\d/:\s.]+)\]: Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)/(\d+)\]\s*eta:\s*([\d:]+)\s*lr:\s*([\d.]+)\s*class_error:\s*([\d.]+)\s*loss:\s*([\d.]+)"
keys = ['Timestamp', 'Epoch', 'Iteration', 'Total Iterations', 'ETA', 'Learning Rate', 'Class Error', 'Loss']

data = []
for log in lines:
    match = re.findall(pattern, log)
    if match:
        values = match[0]
        data.append(dict(zip(keys, values)))

df = pd.DataFrame(data)


iteration = df['Iteration']
class_error = df['Class Error'].astype(float)
loss = df['Loss'].astype(float)

table = pd.DataFrame({'Iteration': iteration, 'Loss': loss, 'Class error':class_error})
table_str = tabulate(table, headers='keys', tablefmt='psql')
print(table_str)

fig, ax1 = plt.subplots()
ax1.plot(iteration, class_error, color='red')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Class Error', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_title('Class Error Trend')
ax1.autoscale(enable=True, axis='y')

plt.savefig('class_error_plot.png')

fig, ax2 = plt.subplots()
ax2.plot(iteration, loss, color='blue')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_title('Loss Trend')
ax2.autoscale(enable=True, axis='y')

plt.savefig('loss_plot.png')
