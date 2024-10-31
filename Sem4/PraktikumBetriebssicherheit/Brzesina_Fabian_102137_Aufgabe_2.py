import graphviz as gv

class BLOCK:

    def __init__(self, name, reliability):
        self.name = name
        self. reliability = reliability

    def printName(self):
        print(self.name)
        return

    def getName(self):
        return self.name
    
    def rel(self):
        return self.reliability
    
class SEQBLOCK:

    def __init__(self, name):
        self.name = name
        self.blocks = []

    def append(self, block):
        self.blocks.append(block)

    def printName(self):
        print(self.name)

    def getName(self):
        return self.name
    
    def rel(self):
        availablity = 1
        for block in self.blocks:
            availablity = availablity * block.rel()
        return 1 - availablity
    
class PARBLOCK:

    def __init__(self, name):
        self.name = name
        self.blocks = []

    def append(self, block):
        self.blocks.append(block)

    def printName(self):
        print(self.name)
        return

    def getName(self):
        return self.name
    
    def rel(self):
        availablity = 1
        for block in self.blocks:
            availablity = availablity * (1 - block.rel())
        return 1 - availablity
    
class GraphPrint:

    def __init__(self, name="Graph"):
        self.topBlock = None
        self.graph = gv.Digraph(name=name,format='png')
        return
    
    def create(self, topBlock):
        self.topBlock = topBlock

        sequence = []
        for block in topBlock.blocks:
            if isinstance(block, BLOCK):
                sequence.append(block)
            else:
                sequence.append(self.createSub(block))

        for i in range(len(sequence)):
            if i > 0:
                firstNode = sequence[i-1]
                secondNode = sequence[i]
                self.graph.edge(sequence[i-1].name, sequence[i].name)

    def createSub(self, topBlock):
    
    def view(self):
        self.graph.view()
        return
    
if __name__ == "__main__":
    E = BLOCK("Eingang", 0.99)
    R1 = BLOCK("Rechner1", 0.99)
    R2 = BLOCK("Rechner2", 0.99)
    A = BLOCK("Ausgang", 0.99)

    seq = SEQBLOCK("Alle")
    par = PARBLOCK("Alle Rechner")

    par.append(R1)
    par.append(R2)

    seq.append(E)
    seq.append(par)
    seq.append(A)

    print(seq.rel())

    #graph = GraphPrint()
    graph = GraphPrint("Zuverlässigkeitsdiagramm")
    graph.create(seq)
    graph.view()
