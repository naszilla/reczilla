import matplotlib.pyplot as plt
from matplotlib import rc
rc("text", usetex=False)
plt.style.use(['science','ieee','no-latex'])

x = [2, 8, 14, 20]
y = [11.4, 11.3, 10.6, 0.0]

plt.plot(x, y, color='blue')
plt.xlabel("# training datasets")
plt.ylabel("Percentage diff. from best")

plt.savefig('fig.png')