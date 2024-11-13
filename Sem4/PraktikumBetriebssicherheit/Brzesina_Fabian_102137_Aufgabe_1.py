import graphviz as gv
import numpy as np
import matplotlib.pyplot as plt

class EVENT:
    
    def __init__(self, name, probability):
        self.name = name
        self.nodes = []
        self.fail = probability

    def add(self, node):
        self.nodes.append(node)
        return
    
    def getFailureProbability(self):
        return self.fail
    
class NOTNODE:
    
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.fail = 0

    def add(self, node):
        self.nodes.append(node)
        return
    
    def getFailureProbability(self):
        self.fail = 1 - self.nodes[0].getFailureProbability()
        return self.fail
    
class ORNODE:
    
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.fail = 0

    def add(self, node):
        self.nodes.append(node)
        return
    
    def getFailureProbability(self):
        newFail = 1
        for node in self.nodes:
            newFail = newFail * (1 - node.getFailureProbability())
        self.fail = 1 - newFail
        return self.fail
    
class ANDNODE:
    
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.fail = 0

    def add(self, node):
        self.nodes.append(node)
        return
    
    def getFailureProbability(self):
        newFail = 1
        for node in self.nodes:
            newFail = newFail * node.getFailureProbability()
        self.fail = newFail
        return self.fail
    
class GraphPrint:

    def __init__(self, name="Graph"):
        self.topNode = None
        self.graph = gv.Digraph(name=name,format='png')
        return
    
    def create(self, topNode):
        self.topNode = topNode
        self.graph.node(self.topNode.name)
        self.getChildNodes(self.topNode)
        return
    
    def getChildNodes(self, node):
        for child in node.nodes:
            self.graph.edge(node.name, child.name)
            self.getChildNodes(child)
        return
    
    def view(self):
        self.graph.view()
        return
    
if __name__ == "__main__":

    # Create the nodes of Picture 1
    A = EVENT("A", 0.01)
    B = EVENT("B", 0.1)
    C = EVENT("C", 0.001)
    D = EVENT("D", 0.01)
    E = EVENT("E", 0.01)
    F = EVENT("F", 0.01)
    G = EVENT("G", 0.1)

    K1 = ANDNODE("K1(AND)")
    K2 = ANDNODE("K2(AND)")
    NOT = NOTNODE("NOT")
    K3 = ORNODE("K3(OR)")
    K4 = ANDNODE("K4(AND)")
    K5 = ORNODE("K5(OR)")

    # Add the nodes to the respective nodes
    K1.add(K2)
    K1.add(NOT)
    #Left side
    K2.add(D)
    K2.add(E)
    K2.add(K4)
    K4.add(K5)
    K4.add(C)
    K5.add(A)
    K5.add(B)
    #Right side
    NOT.add(K3)
    K3.add(F)
    K3.add(G)

    # Calculate the failure probability of the system
    fail = K1.getFailureProbability()
    print(f"The failure probability of the system is: {fail}")

    # Create the graph of Picture 1
    graph = GraphPrint("Graph1")
    graph.create(K1)
    graph.view()


    history = []
    for i in range(1000):
        A = EVENT("A", np.random.normal(0.01, 0.002))
        B = EVENT("B", np.random.normal(0.1, 0.02))
        C = EVENT("C", np.random.normal(0.001, 0.002))
        D = EVENT("D", np.random.normal(0.01, 0.002))
        E = EVENT("E", np.random.normal(0.01, 0.002))
        F = EVENT("F", np.random.normal(0.01, 0.002))
        G = EVENT("G", np.random.normal(0.1, 0.02))

        K1 = ANDNODE("K1(AND)")
        K2 = ANDNODE("K2(AND)")
        NOT = NOTNODE("NOT")
        K3 = ORNODE("K3(OR)")
        K4 = ANDNODE("K4(AND)")
        K5 = ORNODE("K5(OR)")

        # Add the nodes to the respective nodes
        K1.add(K2)
        K1.add(NOT)
        #Left side
        K2.add(D)
        K2.add(E)
        K2.add(K4)
        K4.add(K5)
        K4.add(C)
        K5.add(A)
        K5.add(B)
        #Right side
        NOT.add(K3)
        K3.add(F)
        K3.add(G)

        # print(f"A: {A.getFailureProbability()}")
        # print(f"B: {B.getFailureProbability()}")
        # print(f"C: {C.getFailureProbability()}")
        # print(f"D: {D.getFailureProbability()}")
        # print(f"E: {E.getFailureProbability()}")
        # print(f"F: {F.getFailureProbability()}")
        # print(f"G: {G.getFailureProbability()}")

        # Calculate the failure probability of the system
        fail = K1.getFailureProbability()
        print(f"The failure probability of the system is: {fail}")
        history.append(fail)

    # Create histogram of failure probabilities
    plt.hist(history, bins=150)
    plt.show()