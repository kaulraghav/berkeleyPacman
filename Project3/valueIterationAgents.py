# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(0, self.iterations):
            keyCount = util.Counter()
            for state in self.mdp.getStates():
                if not(self.mdp.isTerminal(state)):
                    maxValue = float("-inf")
                    for action in self.mdp.getPossibleActions(state):
                        value = 0
                        for transition, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                            thd = (self.discount * self.values[transition])
                            snd = (self.mdp.getReward(state, action, transition) + thd)
                            fst = probability * snd
                            value = value + fst
                        maxValue = max(value, maxValue)
                        keyCount[state] = maxValue
                else:
                    keyCount[state] = 0   
            self.values = keyCount

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        for transition, probability in self.mdp.getTransitionStatesAndProbs(state, action):
        	snd = (self.discount * self.getValue(transition))
        	fst = probability * (self.mdp.getReward(state, action, transition) + snd)
        	value = value + fst
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
        	return None
        value = float('-inf')
        for action in self.mdp.getPossibleActions(state):
        	maxQValue = self.computeQValueFromValues(state, action)
        	if maxQValue >= value:
        		value = maxQValue
        		optimalAction = action
        return optimalAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        length = len(states)
        for i in range(0, self.iterations):
            state = states[i%length]
            if not(self.mdp.isTerminal(state)):
                maxValue = float("-inf")
                for action in self.mdp.getPossibleActions(state):
                    value = 0
                    for transition, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        thd = (self.discount * self.values[transition])
                        snd = (self.mdp.getReward(state, action, transition) + thd)
                        fst = probability * snd
                        value = value + fst
                    maxValue = max(value, maxValue)
                self.values[state] = maxValue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pred = {}
        pQueue = util.PriorityQueue()
        
        for i in range(0, self.iterations):
            for state in self.mdp.getStates():
                for action in self.mdp.getPossibleActions(state):
                    for transition, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        if transition in pred:
                            pred[transition].add(state)
                        else:
                            pred[transition] = {state}

        for state in self.mdp.getStates():
            if not(self.mdp.isTerminal(state)):
                currentValue = self.values[state]
                for action in self.mdp.getPossibleActions(state):
                    for transition, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        optimalAction = self.computeActionFromValues(state)
                        maxQValue = self.computeQValueFromValues(state, optimalAction)
                        diff = abs(maxQValue - currentValue)
                        pQueue.update(state, -diff)

        for i in range(0, self.iterations):
            if pQueue.isEmpty():
                break
            state = pQueue.pop()
            if not(self.mdp.isTerminal(state)):
                for action in self.mdp.getPossibleActions(state):
                    for transition, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        optimalAction = self.computeActionFromValues(state)
                        maxQValue = self.computeQValueFromValues(state, optimalAction)
            self.values[state] = maxQValue
            for p in pred[state]:
                if not(self.mdp.isTerminal(p)):
                    currentValue = self.values[p]
                    for action in self.mdp.getPossibleActions(p):
                        for transition, probability in self.mdp.getTransitionStatesAndProbs(p, action):
                            optimalAction = self.computeActionFromValues(p)
                            maxQValue = self.computeQValueFromValues(p, optimalAction)
                diff = abs(maxQValue - currentValue)
                if diff > self.theta:
                    pQueue.update(p, -diff)