# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print('Successor game state is :', successorGameState) 
        newPos = successorGameState.getPacmanPosition()
        #print('New position is :', newPos)
        newFood = successorGameState.getFood()
        #print('New food position is :', newFood) 
        newGhostStates = successorGameState.getGhostStates()
        #print('New ghost position is :', newGhostStates) 
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print('Number of moves that each ghost will remain scared because Pacman has eaten a power pellet :', newScaredTimes)

        "*** YOUR CODE HERE ***"
        score = 0
        previousFood = 0
        currentFood = currentGameState.getFood()
        distance = -1
        
        for width in range(successorGameState.getWalls().width):
            for height in range(successorGameState.getWalls().height):
                if currentFood[width][height] == True:
                    if distance == -1 or distance >= manhattanDistance((width, height), newPos):
                        distance = manhattanDistance((width, height),newPos)
        
        for newGhost in newGhostStates:
            if (manhattanDistance(newGhost.getPosition(), newPos) > 1):
                score = score + 20
            else:
                score = score - 50

        if (currentFood[newPos[0]][newPos[1]] == True):
            score = score + 20

        if distance != -1:
            score = score - distance
        
        if successorGameState.isWin():
            score = score + 500

        return score
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def Minimax(gameState, agent, depth):
            maximum = float("-inf")
            #Checking if depth is reached or game ends 
            if depth >= self.depth or gameState.isLose() == True or gameState.isWin() == True:
                return (self.evaluationFunction(gameState), None)
            else:
                utility = {}
                moves = gameState.getLegalActions(agent)
                for action in moves: 
                    nextState = gameState.generateSuccessor(agent, action)
                    nextAgent = (agent + 1) % gameState.getNumAgents()
                    if nextAgent != 0:
                        valueAction = Minimax(nextState , nextAgent, depth)[0]
                    else:
                        valueAction = Minimax(nextState , nextAgent, depth + 1)[0]
                    utility[action] = valueAction
                
                if agent !=0:
                    return min(utility.values()),[action for action in moves if utility[action] == min(utility.values())]
                else:
                    return max(utility.values()),[action for action in moves if utility[action] == max(utility.values())]
                    

        action = Minimax(gameState, self.index,0)[1][0]
        return action
                        
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        initialAlpha = float("-inf")
        initialBeta = float("inf")
        agents = gameState.getNumAgents()

        def minValue(gameState, agent, depth, alpha, beta):
            v = float('inf')
            currentMove = Directions.STOP
            flag = agent == agents - 1
            ghostMoves = gameState.getLegalActions(agent)
            #Checking if game ends (pacman loses)
            if not ghostMoves or gameState.isLose():
                return self.evaluationFunction(gameState), Directions.STOP
 
            for action in ghostMoves: 
                successor = gameState.generateSuccessor(agent, action)
                if flag: 
                    utility = maxValue(successor, depth + 1, alpha, beta)[0]
                else:
                    utility = minValue(successor, agent + 1, depth, alpha, beta)[0]

                if utility < v:
                    currentMove = action
                    v = utility
                if v < alpha:
                    return v, currentMove
                beta = min(beta, v)

            return v, currentMove

        def maxValue(gameState, depth, alpha, beta):
            v = float('-inf')
            currentMove = Directions.STOP
            pacmanMoves = gameState.getLegalActions()
            #Checking if depth is reached or game ends (pacman wins)
            if depth > self.depth or not pacmanMoves or gameState.isWin():
                return self.evaluationFunction(gameState), Directions.STOP

            for action in pacmanMoves: 
                successor = gameState.generateSuccessor(0, action)
                utility = minValue(successor, 1, depth, alpha, beta)[0]
                if utility > v:
                    currentMove = action
                    v = utility
                if v > beta:
                    return v, currentMove
                alpha = max(alpha, v)

            return v, currentMove

        return maxValue(gameState, 1, initialAlpha, initialBeta)[1]        
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
