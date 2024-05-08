import matplotlib.pyplot as plt

# style is color, marker
line_styles = {
    0: ('b', 'o'),
    1: ('g', '^'),
    2: ('r', 'D')
}

def plot_multiline_graph(title, xlabel, ylabel, lines: list) -> None:
    # lines contains tuple of legend, x, and y in that order
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for number, line in enumerate(lines):
        plt.scatter(line[1], line[2], color=line_styles[number][0], marker=line_styles[number][1], label=line[0])
        plt.plot(line[1], line[2], color=line_styles[number][0])
    plt.legend()
    plt.show()
    

# test main
'''
import numpy as np

x=np.linspace(0, 10, 30)
y1=x*x
y2=2*x
y3=2/x

plot_multiline_graph(title="test graph", xlabel="x-axis", ylabel="y-axis", lines=[
    ("quadratic", x, y1), 
    ("linear", x, y2), 
    ("some otjer thing", x, y3)])
'''