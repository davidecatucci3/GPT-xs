import matplotlib.pyplot as plt

# Data reading
vali = []
train = []
steps = []
val_steps = []
train_steps = []

with open('log/log.txt') as f:
    l = f.readlines()
    for i in l:
        step, type, val = i.split()
        step = int(step)
        steps.append(step)
        if type == 'val':
            vali.append(float(val))
            val_steps.append(step)  # Separate steps for validation
        else:
            train.append(float(val))
            train_steps.append(step)  # Separate steps for training

# Create figure
plt.figure(figsize=(10, 6))

# Plot training data
plt.plot(train_steps, train, label='Training', color='blue', linewidth=2)

# Plot validation data (starting at step 8000 and beyond)
plt.plot(val_steps, vali, label='Validation', color='orange', linewidth=2, linestyle='--')
# Optional: Add markers for validation points
plt.scatter(val_steps, vali, color='orange', s=50, zorder=5)

# Customize plot
plt.title('Training and Validation Progress', fontsize=14, pad=15)
plt.xlabel('Steps', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Ensure x-axis includes all steps
plt.xlim(0, max(steps) * 1.1)

plt.tight_layout()
plt.show()