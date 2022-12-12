#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Multiple plots
fig = plt.figure(figsize=(7, 5), layout="tight")
fig.suptitle("All in One")

# Line graph
ax1 = fig.add_subplot(321)
ax1.plot(y0, "r")
ax1.set(
    yticks=range(0, 1500, 500),
    xticks=range(0, 11, 2),
    xlim=(0, 10),
)

# Men's Height vs Weight
ax2 = fig.add_subplot(322)
ax2.scatter(x1, y1, c="m", marker=".")  # type: ignore
ax2.set_title("Men's Height vs Weight", size="x-small")
ax2.set_xlabel(xlabel=("Height (in)"), size="x-small")
ax2.set_xticks(range(60, 90, 10))
ax2.set_ylabel("Weight (lbs)", size="x-small")

# Expoonential decay of C-14
ax3 = fig.add_subplot(323)
ax3.plot(x2, y2)
ax3.set_title("Exponential Decay of C-14", size="x-small")
ax3.set_xlabel("Time (years)", size="x-small")
ax3.set_ylabel("Fraction Remaining", size="x-small")
ax3.set_yscale("log")
ax3.set_xticks(range(0, 28651, 10000))
ax3.set_xlim(0, 28650)

# Exposure to radioactive elements
ax4 = fig.add_subplot(324)
ax4.plot(x3, y31, "r--", x3, y32, "g")
ax4.set_title("Exponential Decay of Radioactive Elements", size="x-small")
ax4.set_xlabel("Time (years)", size="x-small")
ax4.set_ylabel("Fraction Remaining", size="x-small")
ax4.legend(["C-14", "Ra-226"], fontsize="x-small")
ax4.set_xlim(0, 20000)
ax4.set_ylim(0, 1)

# Project A
ax5 = fig.add_subplot(313)
ax5.hist(student_grades, bins=range(0, 101, 10), edgecolor="black")
ax5.set_xlabel("Grades")
ax5.set_ylabel("Number of Students")
ax5.set_title("Project A")
ax5.set_xticks(range(0, 101, 10))
ax5.set_yticks(range(0, 31, 10))
ax5.set_xlim(0, 100)

plt.show()
