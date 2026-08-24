import matplotlib.pyplot as plt

sizes = [35, 25, 20, 20]
labels = ["Python", "Java", "C", "C++"]

plt.pie(sizes, labels=labels, autopct="%1.1f%%")
plt.title("Programming Language Usage")

plt.show()
