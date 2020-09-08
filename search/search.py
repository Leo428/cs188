# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from game import Directions
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def getPathLst(path):
    if len(path) <= 1:
        return []
    return [direction for pos, direction, cost in path[1:]]

def getPos(node):
    return node[0]

def getDir(node):
    return node[1]

def getCost(node):
    if len(node) != 3:
        return 0
    return node[2]

def getAllPos(path):
    return [getPos(p) for p in path]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # util.raiseNotDefined()
    fringe = util.Stack()
    closed = set()
    path = [[problem.getStartState()]]
    fringe.push(path)

    while not fringe.isEmpty():
        p = fringe.pop()
        node = p[-1]
        nodePos = getPos(node)
        if problem.isGoalState(nodePos):
            return getPathLst(p)
        if nodePos not in closed:
            closed.add(nodePos)
            successors = problem.getSuccessors(nodePos)
            for successor in successors:
                if getPos(successor) not in getAllPos(p):
                    tempP = p.copy()
                    tempP.append(successor)
                    fringe.push(tempP)
    return []



def breadthFirstSearch(problem):
    fringe = util.Queue()
    closed = set()
    path = [[problem.getStartState()]]
    fringe.push(path)

    while not fringe.isEmpty():
        p = fringe.pop()
        node = p[-1]
        nodePos = getPos(node)
        if problem.isGoalState(nodePos):
            return getPathLst(p)
        if nodePos not in closed:
            closed.add(nodePos)
            successors = problem.getSuccessors(nodePos)
            for successor in successors:
                if getPos(successor) not in getAllPos(p):
                    tempP = p.copy()
                    tempP.append(successor)
                    fringe.push(tempP)
    return []
    

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    closed = set()
    path = [[problem.getStartState()]]
    fringe.push(path, 0)

    while not fringe.isEmpty():
        p = fringe.pop()
        node = p[-1]
        nodePos = getPos(node)
        if problem.isGoalState(nodePos):
            return getPathLst(p)
        if nodePos not in closed:
            closed.add(nodePos)
            successors = problem.getSuccessors(nodePos)
            for successor in successors:
                if getPos(successor) not in getAllPos(p):
                    tempP = p.copy()
                    tempS = (getPos(successor), getDir(successor), getCost(p[-1]) + getCost(successor))
                    tempP.append(tempS)
                    fringe.push(tempP, getCost(tempS))
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    closed = set()
    path = [[problem.getStartState()]]
    fringe.push(path, heuristic(problem.getStartState(), problem))

    while not fringe.isEmpty():
        p = fringe.pop()
        node = p[-1]
        nodePos = getPos(node)
        if problem.isGoalState(nodePos):
            return getPathLst(p)
        if nodePos not in closed:
            closed.add(nodePos)
            successors = problem.getSuccessors(nodePos)
            for successor in successors:
                if getPos(successor) not in getAllPos(p):
                    gn = getCost(p[-1]) + getCost(successor)
                    hn = heuristic(getPos(successor), problem)
                    fn = gn + hn
                    tempP = p.copy()
                    tempS = (getPos(successor), getDir(successor), gn)
                    tempP.append(tempS)
                    fringe.push(tempP, fn)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
