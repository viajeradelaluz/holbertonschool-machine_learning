#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

persons = ["Farrah", "Fred", "Felicia"]
fruits = ["apples", "bananas", "oranges", "peaches"]
colors = ["red", "yellow", "#ff8000", "#ffe5b4"]

for i in range(0, len(fruits)):
    plt.bar(
        persons,
        fruit[i],
        bottom=np.sum(fruit[:i], axis=0),
        color=colors[i],
        label=fruits[i],
        width=0.5,
    )

plt.ylabel("Quantity of Fruit")
plt.yticks(range(0, 81, 10))
plt.title("Number of Fruit per Person")
plt.legend()
plt.show()
