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
        prevBlock = None
        for block in topBlock.blocks:
            if isinstance(block, BLOCK):
                self.graph.node(block.name)
            else:
                subgraph = self.createSub(block)
                self.graph.subgraph(subgraph)
                if isinstance(block, SEQBLOCK):
                    if isinstance(prevBlock, BLOCK):
                        self.graph.edge(prevBlock.name, block.blocks[0].name)
                    if isinstance(prevBlock, SEQBLOCK):
                        self.graph.edge(prevBlock.blocks[-1].name, block.blocks[0].name)
                    if isinstance(prevBlock, PARBLOCK):
                        for subBlock in prevBlock.blocks:
                            self.graph.edge(subBlock.name, block.blocks[0].name)
                else:
                    for subBlock in block.blocks:
                        if isinstance(prevBlock, BLOCK):
                            self.graph.edge(prevBlock.name, subBlock.name)
                        if isinstance(prevBlock, SEQBLOCK):
                            self.graph.edge(prevBlock.blocks[-1].name, subBlock.name)
                        if isinstance(prevBlock, PARBLOCK):
                            for subBlock2 in prevBlock.blocks:
                                self.graph.edge(subBlock2.name, subBlock.name)
            if isinstance(block, BLOCK):
                if isinstance(prevBlock, SEQBLOCK):
                    if isinstance(block, BLOCK):
                        self.graph.edge(prevBlock.blocks[-1].name, block.name)
                if isinstance(prevBlock, PARBLOCK):
                    for subBlock in prevBlock.blocks:
                        if isinstance(block, BLOCK):
                            self.graph.edge(subBlock.name, block.name)
                        if isinstance(block, SEQBLOCK):
                            self.graph.edge(subBlock.name, block.blocks[0].name)
            prevBlock = block

        return
    
    def createSub(self, topBlock):
        graph = gv.Digraph(name=f"cluster_{topBlock.name}", graph_attr={'label': topBlock.name})
        if isinstance(topBlock, PARBLOCK):
            for block in topBlock.blocks:
                if isinstance(block, BLOCK):
                    graph.node(block.name)
                else:
                    graph.subgraph(self.createSub(block))
                    if isinstance(block, SEQBLOCK):
                        self.graph.edge(prevBlock.name, block.blocks[0].name)
                    else:
                        for subBlock in block.blocks:
                            self.graph.edge(prevBlock.name, subBlock.name)
        if isinstance(topBlock, SEQBLOCK):
            prevBlock = None
            for block in topBlock.blocks:
                if isinstance(block, BLOCK):
                    graph.node(block.name)
                    if isinstance(prevBlock, BLOCK):
                        graph.edge(prevBlock.name, block.name)
                else:
                    graph.subgraph(self.createSub(block))
                    if isinstance(block, SEQBLOCK):
                        self.graph.edge(prevBlock.name, block.blocks[0].name)
                    else:
                        for subBlock in block.blocks:
                            self.graph.edge(prevBlock.name, subBlock.name)
                prevBlock = block
        return graph
        
    
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

    graph = GraphPrint("Zuverlässigkeitsdiagramm")
    graph.create(seq)
    graph.view()
