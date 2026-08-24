import matplotlib.pyplot as plt

students = ["A", "B", "C", "D"]
marks = [80, 65, 90, 75]

plt.bar(students, marks)
plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")

plt.show()
