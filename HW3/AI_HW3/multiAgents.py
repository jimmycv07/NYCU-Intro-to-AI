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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()
        # print(legalMoves[0])
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        '''
        define a minimax funciton first to get the value recursively. If the agent is Pacman, pick the
        maximun value; If the agent is Ghosts, pick the minimun value. Go through all possible actions
        for every agent in each step.   
        '''
        def minimax(gameStatee, agentIndex, depthh):
            if gameStatee.isWin() or gameStatee.isLose() or depthh==self.depth:
                return self.evaluationFunction(gameStatee)
            elif not agentIndex:
                v=float('-inf')
                for x in gameStatee.getLegalActions(0):
                    v=max(v,minimax(gameStatee.getNextState(0,x),1,depthh))
                return v
            else:
                v=float('inf')
                nextIndex= agentIndex+1
                if nextIndex == gameStatee.getNumAgents():
                    nextIndex=0
                    depthh+=1
                for x in gameStatee.getLegalActions(agentIndex):
                    v=min(v, minimax(gameStatee.getNextState(agentIndex,x),nextIndex, depthh))    
                return v
        '''
        Begin with the top pacman, return the action result in maximun value.
        '''
        maxV= float('-inf')
        maxAct=0
        for x in gameState.getLegalActions(0):
            temp=minimax(gameState.getNextState(0,x),1,0)
            if maxV<temp:
                maxV=temp
                maxAct=x
        return maxAct 
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        '''
        Define the alphabeta function first too. In this function there's 2 addition arguments alpha and beta.
        In picking maximun part, if the value is larger than beta, which means this value wont be chosen by the
        upper minimun part, so we can prun it directly; Same in picking minimum part, if the value is smaller than 
        alpha, we can prun it directly.  
        '''
        def alphabeta(state, agentIndex,d, a, b):
            if state.isWin() or state.isLose() or d==self.depth:
                return self.evaluationFunction(state)
            elif not agentIndex:
                v=float('-inf')
                for x in state.getLegalActions(0):
                    v=max(v,alphabeta(state.getNextState(0,x),1,d,a,b))
                    if v>b:
                        return v
                    a=max(a,v)
                return v
            else :
                nextIndex=agentIndex+1
                if nextIndex==state.getNumAgents():
                    nextIndex=0
                    d+=1
                v=float('inf')
                for x in state.getLegalActions(agentIndex):
                    v=min(v,alphabeta(state.getNextState(agentIndex,x),nextIndex,d,a,b))
                    if v<a:
                        return v
                    b=min(v,b)
                return v
        '''
        Begin with the top pacman as the former part, but update the alpha value for each action.
        '''
        maxV=float('-inf')
        alpha=float('-inf')
        beta=float('inf')
        maxAct=0
        for x in gameState.getLegalActions(0):
            temp=alphabeta(gameState.getNextState(0,x), 1,0,alpha,beta)
            if temp>maxV:
                maxV=temp
                maxAct=x
                alpha=temp
        return maxAct
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        '''
        In this part I also define a expectimax function to get the possible action with maximun value.
        If the agent is ghost, sum the value of all possible action and devide it with the amount of possible
        actions. 
        '''
        def expectimax(state, index,d):
            if state.isWin() or state.isLose() or d==self.depth:
                return self.evaluationFunction(state)
            elif not index:
                v=float('-inf')
                for x in state.getLegalActions(0):
                    v=max(v,expectimax(state.getNextState(0,x),1,d))
                return v
            else:
                nextIndex=index+1
                if nextIndex==state.getNumAgents():
                    nextIndex=0
                    d+=1
                sun=0.0
                for x in state.getLegalActions(index):
                    sun+=expectimax(state.getNextState(index,x), nextIndex, d)
                return sun/ float(len(state.getLegalActions(index)))
        
        maxAct=0
        v=float('-inf')
        for x in gameState.getLegalActions(0):
            temp=expectimax(gameState.getNextState(0,x),1,0)
            if temp>v:
                v=temp
                maxAct=x
        return maxAct
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    '''
    In this better evaluation function part, I take the remain food number, remain capsule number, minimum manhattan
    distance to the active ghost and minimum manhattan distance to the scared ghost into consideration. Weighted these
    attributes and minus them from the current score.
    '''
    pos=currentGameState.getPacmanPosition()
    cap=currentGameState.getCapsules()
    foodNum=currentGameState.getNumFood()
    bonus=0.0
    scaredGhosts, activeGhosts = [], []
    for g in currentGameState.getGhostStates():
        if not g.scaredTimer:
            activeGhosts.append(g)
        else: 
            scaredGhosts.append(g)

    minGhostDis=0
    minSghostDis=0
    if activeGhosts:
        minGhostDis = min(map(lambda g:manhattanDistance(pos,g.getPosition()),activeGhosts))
        
    if scaredGhosts:
        minSghostDis = min(map(lambda g:manhattanDistance(pos, g.getPosition()), scaredGhosts))
        # bonus-=0.5*minSghostDis
        bonus+=1/minSghostDis
    bonus+=3*minGhostDis
    bonus+=5*foodNum
    bonus+=20*len(cap)
    return currentGameState.getScore()-bonus
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
