# Inspired by scikit-learn implementation of the buchheim algorithm aswell as
# py-mag by Bill Mill.
# (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_export.py
# and https://github.com/llimllib/pymag-trees respectively).
from .nodes import LeafNode, DecisionNode
from .decision_tree import DecisionTree
import numpy as np
# Plot an entire tree


def plot_tree(
    tree: DecisionTree,
    impurity=True,
    node_precision=2,
    impurity_precision=3,
    ax=None,
    fontsize=None,
    max_depth=None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.text import Annotation

    if ax is None:
        ax = plt.gca()
    ax.clear()
    ax.set_axis_off()

    my_tree = DrawTree(
        tree.root,
        impurity=impurity,
        node_precision=node_precision,
        impurity_precision=impurity_precision,
    )

    dt = buchheim(my_tree)

    max_x, max_y = dt.max_extents() + 1
    ax_width = ax.get_window_extent().width
    ax_height = ax.get_window_extent().height

    scale_x = ax_width / max_x
    scale_y = ax_height / max_y
    recursive_draw(dt, ax, max_x, max_y, fontsize, max_depth)

    anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]

    renderer = ax.figure.canvas.get_renderer()

    for ann in anns:
        ann.update_bbox_position_size(renderer)

    extents = [ann.get_bbox_patch().get_window_extent() for ann in anns]
    max_width = max([extent.width for extent in extents])
    max_height = max([extent.height for extent in extents])
    scale = min(scale_x / max_width, scale_y / max_height)
    if fontsize is None:
        # get figure to data transform
        # adjust fontsize to avoid overlap
        # get max box width and height
        # width should be around scale_x in axis coordinates
        fontsize = anns[0].get_fontsize() * scale
        for ann in anns:
            ann.set_fontsize(fontsize)

    # Legend of probabilities if it is classification.
    if tree.tree_type == "Classification":
        ax.annotate(
            f"Values: {list(tree.predictor_instance.classes)}",
            (0.01, 1),
            fontsize=fontsize,
            bbox=dict(fc=ax.get_facecolor()),
            ha="center",
            va="center",
            xycoords="axes fraction",
        )

    return anns


def recursive_draw(node, ax, max_x, max_y, fontsize, max_depth, depth=0):
    import matplotlib.pyplot as plt

    kwargs = dict(
        bbox=dict(fc=ax.get_facecolor()),
        ha="center",
        va="center",
        zorder=100 - 10 * depth,
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="<-", edgecolor=plt.rcParams["text.color"]),
    )
    if fontsize is not None:
        kwargs["fontsize"] = fontsize

    # offset things by .5 to center them in plot
    xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

    if max_depth is None or depth <= max_depth:
        if node.parent is None:
            # root
            ax.annotate(node.label, xy, **kwargs)
        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            ax.annotate(node.label, xy_parent, xy, **kwargs)

        for child in node.children:
            recursive_draw(
                child, ax, max_x, max_y, fontsize, max_depth, depth=depth + 1
            )


def get_label(**kwargs):
    node = kwargs["node"]
    impurity_precision = kwargs["impurity_precision"]
    node_precision = kwargs["node_precision"]
    new_line = "\n"
    node_string = ""

    if isinstance(node, DecisionNode):
        node_string += "DecisionNode" + new_line
        node_string += f"X{node.split_idx} <= "
        node_string += str(round(node.threshold,
                           impurity_precision)) + new_line
        if kwargs["impurity"]:
            node_string += "Impurity: "
            node_string += str(round(node.impurity,
                               impurity_precision)) + new_line

    elif isinstance(node, LeafNode):
        node_string += "LeafNode" + new_line
        if kwargs["impurity"]:
            node_string += "Impurity: "
            node_string += str(round(node.impurity,
                               impurity_precision)) + new_line
        node_string += "Samples: "
        node_string += str(round(node.weighted_samples,
                           impurity_precision)) + new_line
        node_string += "Value: "
        if len(node.value) == 1:
            node_string += str(round(node.value[0], node_precision))
        else:
            node_value_string = "\n ["
            value_length = len(node.value)
            n_vals_per_line = max(
                value_length / 3,
                4)  # Number of values per line
            for i in range(value_length):
                node_value_string += str(round(node.value[i], node_precision))
                if (i + 1) % n_vals_per_line == 0 and i != value_length - 1:
                    node_value_string += new_line
                elif i != value_length - 1:
                    node_value_string += ", "
            node_value_string += "]"
            node_string += node_value_string
    return node_string


class DrawTree(object):
    def __init__(self, node, parent=None, depth=0, number=1, **kwargs):
        self.x = -1.0
        self.y = depth
        self.node = node
        lst = []
        if isinstance(node, DecisionNode):
            # add left child first

            if node.left_child is not None:
                lst.append(
                    DrawTree(
                        node.left_child,
                        self,
                        depth + 1,
                        number=1,
                        **kwargs))
            if node.right_child is not None:
                lst.append(
                    DrawTree(
                        node.right_child,
                        self,
                        depth + 1,
                        number=2,
                        **kwargs))
        self.children = lst
        self.parent = parent
        self.thread = None
        self.mod = 0
        self.ancestor = self
        self.change = self.shift = 0
        self._lmost_sibling = None
        self.label = get_label(node=node, **kwargs)
        # this is the number of the node in its group of siblings 1..n
        self.number = number

    def left(self):
        return self.thread or len(self.children) and self.children[0]

    def right(self):
        return self.thread or len(self.children) and self.children[-1]

    def lbrother(self):
        n = None
        if self.parent:
            for node in self.parent.children:
                if node == self:
                    return n
                else:
                    n = node
        return n

    def get_lmost_sibling(self):
        if not self._lmost_sibling and self.parent and self != self.parent.children[0]:
            self._lmost_sibling = self.parent.children[0]
        return self._lmost_sibling

    lmost_sibling = property(get_lmost_sibling)

    def __str__(self):
        return "x=%s mod=%s" % (self.x, self.mod)

    def __repr__(self):
        return self.__str__()

    def max_extents(self):
        extents = [c.max_extents() for c in self.children]
        extents.append((self.x, self.y))
        return np.max(extents, axis=0)


def buchheim(tree):
    dt = firstwalk(tree)
    min = second_walk(dt)
    if min < 0:
        third_walk(dt, -min)
    return dt


def third_walk(tree, n):
    tree.x += n
    for c in tree.children:
        third_walk(c, n)


def firstwalk(v, distance=1.0):
    if len(v.children) == 0:
        if v.lmost_sibling:
            v.x = v.lbrother().x + distance
        else:
            v.x = 0.0
    else:
        default_ancestor = v.children[0]
        for w in v.children:
            firstwalk(w)
            default_ancestor = apportion(w, default_ancestor, distance)
        execute_shifts(v)

        midpoint = (v.children[0].x + v.children[-1].x) / 2

        # ell = v.children[0]
        # arr = v.children[-1]
        w = v.lbrother()
        if w:
            v.x = w.x + distance
            v.mod = v.x - midpoint
        else:
            v.x = midpoint
    return v


def apportion(v, default_ancestor, distance):
    w = v.lbrother()
    if w is not None:
        # in buchheim notation:
        # i == inner; o == outer; r == right; l == left; r = +; l = -
        vir = vor = v
        vil = w
        vol = v.lmost_sibling
        sir = sor = v.mod
        sil = vil.mod
        sol = vol.mod
        while vil.right() and vir.left():
            vil = vil.right()
            vir = vir.left()
            vol = vol.left()
            vor = vor.right()
            vor.ancestor = v
            shift = (vil.x + sil) - (vir.x + sir) + distance
            if shift > 0:
                move_subtree(ancestor(vil, v, default_ancestor), v, shift)
                sir = sir + shift
                sor = sor + shift
            sil += vil.mod
            sir += vir.mod
            sol += vol.mod
            sor += vor.mod
        if vil.right() and not vor.right():
            vor.thread = vil.right()
            vor.mod += sil - sor
        else:
            if vir.left() and not vol.left():
                vol.thread = vir.left()
                vol.mod += sir - sol
            default_ancestor = v
    return default_ancestor


def move_subtree(wl, wr, shift):
    subtrees = wr.number - wl.number
    wr.change -= shift / subtrees
    wr.shift += shift
    wl.change += shift / subtrees
    wr.x += shift
    wr.mod += shift


def execute_shifts(v):
    shift = change = 0
    for w in v.children[::-1]:
        w.x += shift
        w.mod += shift
        change += w.change
        shift += w.shift + change


def ancestor(vil, v, default_ancestor):
    # the relevant text is at the bottom of page 7 of
    # "Improving Walker's Algorithm to Run in Linear Time" by Buchheim et al, (2002)
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8757&rep=rep1&type=pdf
    if vil.ancestor in v.parent.children:
        return vil.ancestor
    else:
        return default_ancestor


def second_walk(v, m=0, depth=0, min=None):
    v.x += m
    v.y = depth

    if min is None or v.x < min:
        min = v.x

    for w in v.children:
        min = second_walk(w, m + v.mod, depth + 1, min)

    return min
