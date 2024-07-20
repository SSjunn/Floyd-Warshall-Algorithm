import tkinter as tk
from tkinter import messagebox
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch

def create_nearby_connected_graph(node_size, radius, degree, bidirectional):
    positions = {}
    edges = []
    adj_matrix = np.full((node_size, node_size), np.inf)
    np.fill_diagonal(adj_matrix, 0)
    degree_count = [0] * node_size

    sqrt_size = int(np.ceil(np.sqrt(node_size)))
    for i in range(node_size):
        x = (i % sqrt_size) / sqrt_size + random.uniform(-0.05, 0.05)
        y = (i // sqrt_size) / sqrt_size + random.uniform(-0.05, 0.05)
        positions[i] = (x, y)

    for i in range(node_size - 1):
        if degree_count[i] < degree and degree_count[i + 1] < degree:
            weight = random.randint(1, 30)
            edges.append((i, i + 1, weight))
            adj_matrix[i][i + 1] = weight
            if bidirectional:
                adj_matrix[i + 1][i] = weight
                edges.append((i + 1, i, weight))
            degree_count[i] += 1
            degree_count[i + 1] += 1

    for i in range(node_size):
        for j in range(i + 2, node_size):
            if np.linalg.norm(np.array(positions[i]) - np.array(positions[j])) < radius and degree_count[i] < degree and degree_count[j] < degree:
                weight = random.randint(1, 30)
                edges.append((i, j, weight))
                adj_matrix[i][j] = weight
                if bidirectional:
                    adj_matrix[j][i] = weight
                    edges.append((j, i, weight))
                degree_count[i] += 1
                degree_count[j] += 1

    return positions, edges, adj_matrix

def floyd_warshall(adj_matrix):
    dist_matrix = adj_matrix.copy()
    node_size = len(adj_matrix)
    next_node = np.zeros((node_size, node_size), dtype=int) - 1

    for i in range(node_size):
        for j in range(node_size):
            if i != j and dist_matrix[i][j] != np.inf:
                next_node[i][j] = j

    for k in range(node_size):
        for i in range(node_size):
            for j in range(node_size):
                if dist_matrix[i][j] > dist_matrix[i][k] + dist_matrix[k][j]:
                    dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
                    next_node[i][j] = next_node[i][k]

    return dist_matrix, next_node

def reconstruct_path(start, end, next_node):
    if next_node[start][end] == -1:
        return []
    path = [start]
    while start != end:
        start = next_node[start][end]
        path.append(start)
    return path

class FloydWarshallApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Floyd-Warshall Algorithm Visualization")

        self.input_frame = tk.Frame(root)
        self.input_frame.pack(pady=20)

        self.start_label = tk.Label(self.input_frame, text="Start Node:")
        self.start_label.pack(side=tk.LEFT)
        self.start_entry = tk.Entry(self.input_frame)
        self.start_entry.pack(side=tk.LEFT)

        self.end_label = tk.Label(self.input_frame, text="End Node:")
        self.end_label.pack(side=tk.LEFT)
        self.end_entry = tk.Entry(self.input_frame)
        self.end_entry.pack(side=tk.LEFT)

        self.node_size_label = tk.Label(self.input_frame, text="Node Size:")
        self.node_size_label.pack(side=tk.LEFT)
        self.node_size_entry = tk.Entry(self.input_frame)
        self.node_size_entry.pack(side=tk.LEFT)

        self.degree_label = tk.Label(self.input_frame, text="Max Degree:")
        self.degree_label.pack(side=tk.LEFT)
        self.degree_entry = tk.Entry(self.input_frame)
        self.degree_entry.pack(side=tk.LEFT)

        self.bidirectional_var = tk.IntVar()
        self.bidirectional_check = tk.Checkbutton(
            self.input_frame, text="Bidirectional Edges", variable=self.bidirectional_var,
            command=self.visualize_shortest_path)
        self.bidirectional_check.pack(side=tk.LEFT)

        self.submit_button = tk.Button(self.input_frame, text="Visualize", command=self.visualize_shortest_path)
        self.submit_button.pack(side=tk.LEFT)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(pady=20)

    def visualize_shortest_path(self):
        try:
            self.start_node = int(self.start_entry.get())
            self.end_node = int(self.end_entry.get())
            self.node_size = int(self.node_size_entry.get())
            self.degree = int(self.degree_entry.get())
            self.radius = 0.2
            self.bidirectional = self.bidirectional_var.get() == 1

            self.positions, self.edges, self.adj_matrix = create_nearby_connected_graph(self.node_size, self.radius, self.degree, self.bidirectional)
            self.dist_matrix, self.next_node = floyd_warshall(self.adj_matrix)
            self.shortest_path_length = self.dist_matrix[self.start_node][self.end_node]
            self.shortest_path = reconstruct_path(self.start_node, self.end_node, self.next_node)

            if not self.shortest_path:
                raise IndexError

            # 모든 경로의 평균 가중치 계산
            total_weight = 0
            edge_count = 0
            for i in range(self.node_size):
                for j in range(self.node_size):
                    if self.adj_matrix[i][j] != np.inf and i != j:
                        total_weight += self.adj_matrix[i][j]
                        edge_count += 1
            self.average_weight = total_weight / edge_count if edge_count > 0 else 0

            self.update_graph()

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid integers for nodes, size, and degree")
        except IndexError:
            messagebox.showerror("No Path Error", "No path exists between the selected nodes")

    def update_graph(self):
        if not hasattr(self, 'positions'):
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        bidirectional = self.bidirectional_var.get() == 1
        for (i, j, weight) in self.edges:
            arrow = FancyArrowPatch((self.positions[i][0], self.positions[i][1]),
                                    (self.positions[j][0], self.positions[j][1]),
                                    arrowstyle='-|>' if not bidirectional else '<|-|>',
                                    color='gray', mutation_scale=10, alpha=0.5)
            ax.add_patch(arrow)

        for i in range(self.node_size):
            ax.scatter(self.positions[i][0], self.positions[i][1], c='skyblue', s=50)

        for i in range(len(self.shortest_path) - 1):
            arrow = FancyArrowPatch((self.positions[self.shortest_path[i]][0], self.positions[self.shortest_path[i]][1]),
                                    (self.positions[self.shortest_path[i + 1]][0], self.positions[self.shortest_path[i + 1]][1]),
                                    arrowstyle='-|>', color='r', mutation_scale=15)
            ax.add_patch(arrow)
            if bidirectional:
                arrow = FancyArrowPatch((self.positions[self.shortest_path[i + 1]][0], self.positions[self.shortest_path[i + 1]][1]),
                                        (self.positions[self.shortest_path[i]][0], self.positions[self.shortest_path[i]][1]),
                                        arrowstyle='-|>', color='r', mutation_scale=15)
                ax.add_patch(arrow)
            ax.scatter(self.positions[self.shortest_path[i]][0], self.positions[self.shortest_path[i]][1], c='red', s=100)
        ax.scatter(self.positions[self.shortest_path[-1]][0], self.positions[self.shortest_path[-1]][1], c='red', s=100)

        for i in self.shortest_path:
            ax.text(self.positions[i][0], self.positions[i][1], str(i), color="black", fontsize=12)

        ax.set_title(f"Graph with {self.node_size} nodes\nAverage weight of {'bidirectional' if bidirectional else 'unidirectional'} paths: {self.average_weight:.2f}")
        fig.text(0.5, 0.01, f"Shortest path length: {self.shortest_path_length}", ha='center', transform=fig.transFigure)
        fig.text(0.5, 0.03, f"Shortest path: {self.shortest_path}", ha='center', transform=fig.transFigure)

        self.display_plot(fig)

    def display_plot(self, fig):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = FloydWarshallApp(root)
    root.mainloop()
