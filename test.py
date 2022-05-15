# %%
from nn import RNN, Parameter, Linear, LSTM
from tensor import Graph, Tensor, randn, rand

Graph.clear()
rnn = LSTM(8, 32, batch_first=True)
x = randn(10, 8, 8)

y = rnn(x).sum()
# %%
from graphviz import Digraph

g = Digraph(graph_attr={"rankdir": "LR"}, format='png')
# BFS画图，会画出对输出有影响的节点，不管是否需要求导


def get_name(node):
    if isinstance(node, Parameter):
        return "Param {:}".format(node.shape)
    return str(node).split(", ")[-1][:-1]


node_queue = [y]
explored = set()
while len(node_queue) > 0:
    node = node_queue.pop()
    explored.add(node)
    g.node(
        str(Graph.node_list.index(node)),
        label=get_name(node),
    )
    for last in node.last:
        if last not in explored and last.requires_grad:
            g.edge(str(Graph.node_list.index(last)),
                   str(Graph.node_list.index(node)))
            node_queue.insert(0, last)
g
# %%
y.backward()
# %%