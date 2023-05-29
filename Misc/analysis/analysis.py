import pandas as pd
import re
import matplotlib.pyplot as plt

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

print(df)

df = df.sort_values('Iteration')

iteration = df['Iteration']
class_error = df['Class Error']
loss = df['Loss']

fig, ax1 = plt.subplots()
ax1.plot(iteration, class_error, color='red')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Class Error', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_yscale('log')
ax1.set_title('Class Error Trend (Logarithmic Scale)')

plt.savefig('class_error_plot.png')

fig, ax2 = plt.subplots()
ax2.plot(iteration, loss, color='blue')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_yscale('log')
ax2.set_title('Loss Trend (Logarithmic Scale)')

plt.savefig('loss_plot.png')

plt.show()
