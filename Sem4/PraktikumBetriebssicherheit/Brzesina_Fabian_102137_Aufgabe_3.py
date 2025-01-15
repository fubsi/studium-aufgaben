import graphviz as gv

class STATE:
    def __init__(self, name, num):
        self.name = name
        self.num = num 
        self.qstates = []
        self.v = 0.0  # gesamtbelohnung

    def add(self, qstate):
        self.qstates.append(qstate)

    def utility(self):
        self.v = max([q.utility() for q in self.qstates])
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
        self.q = sum([t.prop * (t.reward + self.gamma * t.destination.v) for t in self.transitions])
        return self.q

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
        self.states = []
        self.qstates = []
        self.transitions = []

    def state(self, snode):
        self.states.append(snode)

    def qstate(self, qnode):
        self.qstates.append(qnode)

    def transition(self, transition):
        self.transitions.append(transition)

    def show(self):
        G = GraphPrint(self)
        G.create()
        G.show()

    def printnodes(self):
        print(f"{'-'*100}")
        print('States:')
        for s in self.states:
            print(s.name, end=' | ')
        print(f"\n{'-'*100}")
        print('Q-States:')
        for q in self.qstates:
            print(q.name, end=' | ')
        print(f"\n{'-'*100}")
        print('Transitions:')
        for t in self.transitions:
            print(t.name, end=' | ')
        print(f"\n{'-'*100}")

    def utility(self):
        # Value Iteration
        for _ in range(100):  # Adjust the number of iterations as needed
            for state in self.states:
                state.utility()

        print('Utility:')
        for s in self.states:
            print(f"{s.name}:{s.utility()}")
        for q in self.qstates:
            print(f"{q.name}:{q.q}")
        print(f"{'-'*100}")

        print('Optimal Strategy for States:')
        for s in self.states:
            print(f"{s.name} mit {self.optimalOfState(s).action} -> {self.optimalOfState(s).name}")
        print(f"{'-'*100}")

        print('Optimal Strategy:')
        nextState = self.states[0]

        for i in range(10):
            nextOptimal = self.optimalOfState(nextState)
            print(f"Step {i+1}: {nextState.name} mit {nextOptimal.action} -> {nextOptimal.name}", end='')
            for t in self.transitions:
                if t.source.state == nextState and t.source == nextOptimal and t.source.action == nextOptimal.action:
                    nextState = t.destination
            print(f" -> {nextState.name}")

    def optimalOfState(self, state):
        maxQ = max([q.utility() for q in state.qstates])
        for q in state.qstates:
            if q.utility() == maxQ:
                return q

class GraphPrint:

    def __init__(self, objMDP, name="Graph"):
        self.MDP = objMDP
        self.topBlock = None
        self.graph = gv.Digraph(name=name, format='png')

    def create(self):
        for state in self.MDP.states:
            self.graph.node(state.name, shape='triangle')
        for qstate in self.MDP.qstates:
            self.graph.node(qstate.name, shape='circle')
            self.graph.edge(qstate.state.name, qstate.name, label=qstate.action, color='red')
        for transition in self.MDP.transitions:
            self.graph.edge(transition.source.name, transition.destination.name, label=f"{str(transition.prop)}")

    def show(self):
        self.graph.view()

if __name__ == '__main__':
    M = MDP('Aufgabe3')

    #Zustände
    S_PARKEN_OI = STATE('Parken o. Inspek.', 1)
    S_PARKEN_MI = STATE('Parken m. Inspek.', 2)
    S_FAHREN_OI = STATE('Fahren o. Inspek.', 3)
    S_FAHREN_MI = STATE('Fahren m. Inspek.', 4)
    S_INSPEKTION = STATE('Inspektion', 5)

    #Q-Zustände
    Q_POI_a = QSTATE('Q_POI_a', S_PARKEN_OI, 'betreiben') #nach Aktion A
    S_PARKEN_OI.add(Q_POI_a)

    Q_FOI_a = QSTATE('Q_FOI_a', S_FAHREN_OI, 'betreiben') #nach Aktion A
    Q_FOI_b = QSTATE('Q_FOI_b', S_FAHREN_OI, 'warten') #nach Aktion B (Inspektion)
    S_FAHREN_OI.add(Q_FOI_a)
    S_FAHREN_OI.add(Q_FOI_b)

    Q_I_a = QSTATE('Q_I_a', S_INSPEKTION, 'warten') #nach Aktion A
    S_INSPEKTION.add(Q_I_a)

    Q_FMI_a = QSTATE('Q_FMI_a', S_FAHREN_MI, 'betreiben') #nach Aktion A
    Q_FMI_b = QSTATE('Q_FMI_b', S_FAHREN_MI, 'warten') #nach Aktion B (Inspektion)
    S_FAHREN_MI.add(Q_FMI_a)
    S_FAHREN_MI.add(Q_FMI_b)

    Q_PMI_a = QSTATE('Q_PMI_a', S_PARKEN_MI, 'betreiben') #nach Aktion A
    S_PARKEN_MI.add(Q_PMI_a)

    #Transitionen
    T_POI_a = TRANSITION('T_POI_a', Q_POI_a, S_PARKEN_OI, 0.6, 50) 
    T_POI_b = TRANSITION('T_POI_b', Q_POI_a, S_FAHREN_OI, 0.4, 50)
    Q_POI_a.add(T_POI_a)
    Q_POI_a.add(T_POI_b)

    T_FOI_a = TRANSITION('T_FOI_a', Q_FOI_a, S_PARKEN_OI, 0.6, 50)
    T_FOI_b = TRANSITION('T_FOI_b', Q_FOI_a, S_FAHREN_OI, 0.4, 50)
    T_FOI_c = TRANSITION('T_FOI_c', Q_FOI_b, S_INSPEKTION, 0.006, -200) # 1h/1Woche = 1h/168h = 1/168 = 0.006
    Q_FOI_a.add(T_FOI_a)
    Q_FOI_a.add(T_FOI_b)
    Q_FOI_b.add(T_FOI_c)

    T_I_a = TRANSITION('T_I_a', Q_I_a, S_INSPEKTION, 0.75, -20) # 1 - 1h/4h = 3h/4h = 3/4 = 0.75
    T_I_b = TRANSITION('T_I_b', Q_I_a, S_FAHREN_MI, 0.25, 500) # 1h/4h = 1/4 = 0.25
    Q_I_a.add(T_I_a)
    Q_I_a.add(T_I_b)

    T_FMI_a = TRANSITION('T_FMI_a', Q_FMI_a, S_FAHREN_OI, 0.00057, 50) # 1h/2Jahre = 1h/17520h = 1/17520 = 5,7*10^-5 = 0.000057
    T_FMI_b = TRANSITION('T_FMI_b', Q_FMI_a, S_FAHREN_MI, 0.399943, 50) # 1-1h/2Jahre-0.6 = 1-1h/17520h-0.6 = 1-1/17520 = 1-5.7*10^-5 -0.6 = 0.999943 - 0.6 = 0.399943
    T_FMI_c = TRANSITION('T_FMI_c', Q_FMI_a, S_PARKEN_MI, 0.6, 50)
    T_FMI_d = TRANSITION('T_FMI_d', Q_FMI_b, S_INSPEKTION, 0.00057, -20) #1h/2Jahre
    Q_FMI_a.add(T_FMI_a)
    Q_FMI_a.add(T_FMI_b)
    Q_FMI_a.add(T_FMI_c)
    Q_FMI_b.add(T_FMI_d)

    T_PMI_a = TRANSITION('T_PMI_a', Q_PMI_a, S_PARKEN_OI, 0.00057, 50) #1h/2Jahre
    T_PMI_b = TRANSITION('T_PMI_b', Q_PMI_a, S_PARKEN_MI, 0.59943, 50) #1h/2Jahre-0.4 = 0.999943-0.4 = 0.59943
    T_PMI_c = TRANSITION('T_PMI_c', Q_PMI_a, S_FAHREN_MI, 0.4, 50)
    Q_PMI_a.add(T_PMI_a)
    Q_PMI_a.add(T_PMI_b)
    Q_PMI_a.add(T_PMI_c)

    #Zustände hinzufügen
    M.state(S_PARKEN_OI)
    M.state(S_PARKEN_MI)
    M.state(S_FAHREN_OI)
    M.state(S_FAHREN_MI)
    M.state(S_INSPEKTION)

    #Q-Zustände hinzufügen
    M.qstate(Q_POI_a)
    
    M.qstate(Q_FOI_a)
    M.qstate(Q_FOI_b)

    M.qstate(Q_I_a)

    M.qstate(Q_FMI_a)
    M.qstate(Q_FMI_b)

    M.qstate(Q_PMI_a)

    #Transitionen hinzufügen
    M.transition(T_POI_a)
    M.transition(T_POI_b)

    M.transition(T_FOI_a)
    M.transition(T_FOI_b)
    M.transition(T_FOI_c)

    M.transition(T_I_a)
    M.transition(T_I_b)
    
    M.transition(T_FMI_a)
    M.transition(T_FMI_b)
    M.transition(T_FMI_c)
    M.transition(T_FMI_d)

    M.transition(T_PMI_a)
    M.transition(T_PMI_b)
    M.transition(T_PMI_c)

    M.printnodes()

    #Graphen erstellen
    M.show()

    #Utility berechnen
    M.utility()