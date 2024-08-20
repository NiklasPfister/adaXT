from . import DecisionTree, LeafNode, DecisionNode
# Plot an entire tree


def plot_tree(
    tree: DecisionTree,
    impurity=True,
    node_ids=False,
    precision=3,
    ax=None,
) -> None:
    """
    Generates the tree in a subplot of plt. To show the plot,
    the user needs to call matplotlib.pyplot.show().

    Parameters
    ----------
    tree : DecisionTree
        the tree to plot
    """
    plotter = DecisionTreePlotter(
        impurity=impurity,
        node_ids=node_ids,
        precision=precision,
    )
    plotter.plot(tree=tree, ax=ax)


class DecisionTreePlotter():

    def __init__(
        self,
        impurity=True,
        node_ids=False,
        precision=3,
    ) -> None:
        self.impurity = impurity
        self.node_ids = node_ids
        self.precision = precision
        self.depth_distance = 10
        self.width_distance = 10

    def plot_leaf_node(self, node: LeafNode, position: tuple):
        text = f"""
                Leaf Node
                Impurity: {node.impurity:.3f}
                weighted_samples: {node.weighted_samples}
                value: {['%.2f' % x for x in node.value]}
                """
        self.ax.text(
            position[0],
            position[1],
            "\n".join(line.strip() for line in text.splitlines()),
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="black"),
            wrap=True,
        )
        print(node.value)

    def plot_decision_node(self, node: DecisionNode, position: tuple):
        text = f"""
                Decision Node
                x{node.split_idx} <= {node.threshold:.3f}
                Impurity: {node.impurity:.3f}
                """
        self.ax.text(
            position[0],
            position[1],
            "\n".join(line.strip() for line in text.splitlines()),
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="black"),
            wrap=True,
        )

    def calculate_node_positions(
            self,
            node: DecisionNode | LeafNode | None,
            x: float,
            y: float):
        if node is None:
            return {}

        dx = 1
        dy = 1
        if isinstance(node, DecisionNode):
            left_positions = self.calculate_node_positions(
                node.left_child, 2 * x - dx, y - dy)
            right_positions = self.calculate_node_positions(
                node.right_child, 2 * x + dx, y - dy)
        else:
            left_positions = self.calculate_node_positions(
                None, 2 * x - dx, y - dy)
            right_positions = self.calculate_node_positions(
                None, 2 * x + dx, y - dy)

        position = (x, y)
        node_positions = {**left_positions, **right_positions, node: position}
        return node_positions

    def plot_node(self, node):
        """
        Helper function used to plot each node of a DecisionTree

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axes to plot on
        node : Node
            node type of a tree
        node_positions : tuple
            (left_child position, right_child position, nodes own position)
        """
        if node is None:
            return

        position = self.node_positions[node]

        # Draw the node box
        if isinstance(node, LeafNode):
            self.plot_leaf_node(
                node, position
            )
        else:
            self.plot_decision_node(
                node, position
            )

        # Draw edges and child nodes recursively
        if isinstance(node, DecisionNode):
            if node.left_child is not None:
                self.ax.plot(
                    [position[0], self.node_positions[node.left_child][0]],
                    [position[1], self.node_positions[node.left_child][1]],
                    color="black",
                )
                self.plot_node(node.left_child)
            if node.right_child is not None:
                self.ax.plot(
                    [position[0], self.node_positions[node.right_child][0]],
                    [position[1], self.node_positions[node.right_child][1]],
                    color="black",
                )
                self.plot_node(node.right_child)

    def plot(self, tree: DecisionTree, ax=None) -> None:
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        self.ax = ax
        self.node_positions = self.calculate_node_positions(tree.root, 0, 0)
        self.plot_node(tree.root)


def print_tree(tree: DecisionTree):
    queue = []
    queue.append(tree.root)
    while len(queue) > 0:
        node = queue.pop()
        if node:
            print(f"Depth: {node.depth}")
            print(f"Impurity: {node.impurity}")
            print(f"samples: {node.n_samples}")
            if isinstance(node, LeafNode):
                print(f"LEAF WITH VAL: {node.value}")
            else:
                print(f"Decision WITH x{node.split_idx} <= {node.threshold}")
            print("")  # spacing
            if isinstance(node, DecisionNode):
                queue.append(node.left_child)
                queue.append(node.right_child)
