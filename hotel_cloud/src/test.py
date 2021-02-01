import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_, ax = plt.subplots(figsize=(10, 8))
ax.plot([1,2,3], label="trash")
plt.show()