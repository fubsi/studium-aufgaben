#Folie 20 Kapitel 12 soll helfen

class STATE:
    def __init__(self, name, num):
        self.name = name
        self.num = num
        self.qstates = []
        self.v = 0.0        #gesamtbelohnung
    
    def add(self, qstate):
        self.qstates.append(qstate)

    def utility(self):
        return self.v
    
class QSTATE:
    def __init__(self, name, state, action):
        if not isinstance(state, STATE):
            exit(0)
        self.name = name
        self.state = state
        self.action = action
        self.transitions = []
        self.q = 0.0
        self.gamma = 0.9
        
    def add(self, transition):
        self.transitions.append(transition)
    
    def utility(self):
        ...

class TRANSITION:
    def __init__(self, name, source, destination, prop, reward):
        if not isinstance(source, QSTATE) or not isinstance(destination, STATE):
            exit(0)
        self.name = name
        self.source = source
        self.destination = destination
        self.prop = prop
        self.reward = reward

class MDP:
    def __init__(self, name):
        self.name = name

    def state(self, snode):
        ...
    
    def qstate(self, qnode):
        ...

    def transition(self, transition):
        ...
    
    def show(self):
        ...

    def printnodes(self):
        ...
    
    def utility(self):
        ...

if __name__ == '__main__':
    M = MDP('Aufgabe3')