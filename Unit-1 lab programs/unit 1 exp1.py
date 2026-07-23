# pip install matplotlib

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 15, 8, 20, 18]

plt.plot(x, y)
plt.title("Simple Line Graph")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.show()
