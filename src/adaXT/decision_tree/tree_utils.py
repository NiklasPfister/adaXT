from . import DecisionTree, LeafNode, DecisionNode
import textwrap
# Plot an entire tree


def plot_tree(tree: DecisionTree):
    """
    Generates the tree in a subplot of plt. To show the plot,
    the user needs to call matplotlib.pyplot.show().

    Parameters
    ----------
    tree : DecisionTree
        the tree to plot

    Returns
    -------
    matplotlib.figure.Figure
        the figure of the subplot
    matplotlib.axes.Axes
        the axes of the subplot
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    node_positions = calculate_node_positions(tree.root, x=0, y=0)
    plot_node(ax, tree.root, node_positions)
    ax.axis("off")
    return fig, ax


# Plot a node


def plot_node(ax, node: LeafNode | DecisionNode, node_positions: tuple):
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

    position = node_positions[node]

    # Draw the node box
    if isinstance(node, LeafNode):
        ax.text(
            position[0],
            position[1],
            textwrap.dedent(
                f"""\
            Leaf Node\n\
            Impurity: {node.impurity:.3f}\n\
            weighted_samples: {node.weighted_samples}\n\
            value: {['%.2f' % x for x in node.value]}
            """
            ),
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="black"),
        )
    else:
        ax.text(
            position[0],
            position[1],
            textwrap.dedent(
                f"""\
                Decision Node\n\
                x{node.split_idx} <= {node.threshold:.3f}\n\
                Impurity: {node.impurity:.3f}\n\
                samples: {node.n_samples}
                """
            ),
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="black"),
        )

    # Draw edges and child nodes recursively
    if isinstance(node, DecisionNode):
        if node.left_child is not None:
            ax.plot(
                [position[0], node_positions[node.left_child][0]],
                [position[1], node_positions[node.left_child][1]],
                color="black",
            )
            plot_node(ax, node.left_child, node_positions)
        if node.right_child is not None:
            ax.plot(
                [position[0], node_positions[node.right_child][0]],
                [position[1], node_positions[node.right_child][1]],
                color="black",
            )
            plot_node(ax, node.right_child, node_positions)


# Calculate where to add nodes when plotting a tree


def calculate_node_positions(
        node: LeafNode | DecisionNode,
        x: float,
        y: float):
    if node is None:
        return {}

    dx = 1
    dy = 1
    if isinstance(node, DecisionNode):
        left_positions = calculate_node_positions(
            node.left_child, 2 * x - dx, y - dy)
        right_positions = calculate_node_positions(
            node.right_child, 2 * x + dx, y - dy)
    else:
        left_positions = calculate_node_positions(None, 2 * x - dx, y - dy)
        right_positions = calculate_node_positions(None, 2 * x + dx, y - dy)

    position = (x, y)

    node_positions = {**left_positions, **right_positions, node: position}

    return node_positions


# Function to print the information of a tree


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
