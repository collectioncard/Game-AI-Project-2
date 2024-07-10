from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 1000
explore_faction = 2.

# Done
def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exists,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node

    """
    # Loop through the nodes
    while node.untried_actions == [] and node.child_nodes != {}:
        currentBestUcb = -9999  # That's gotta be low enough right? I can't possibly make a move *that* bad
        bestChild = None

        # Check every child node and see which one is the best route to take
        for action, child in node.child_nodes.items():
            # Calculate the UCB value and replace it if it's better than the current best
            ucbValue = ucb(child, bot_identity == board.current_player(state))

            if ucbValue > currentBestUcb:
                currentBestUcb = ucbValue
                bestChild = child

        # Continue the search from that best child by replacing the node and state
        if bestChild is not None:
            node = bestChild
            state = board.next_state(state, bestChild.parent_action)

    return node, state


#Done
def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    # Let's first check to ensure that the node is not terminal (has untried actions)
    if not node.untried_actions:
        return None, state

    # Get the next untried action and try to expand upon it
    actionToExpand = node.untried_actions.pop()

    # Get the new game state to be used for the new child node
    newGameState = board.next_state(state, actionToExpand)

    # Create the new child node to be added to the given nodes's child nodes
    newChildNode = MCTSNode(parent=node, parent_action=actionToExpand, action_list=board.legal_actions(newGameState))
    node.child_nodes[actionToExpand] = newChildNode

    return newChildNode, newGameState


# Done
def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """

    while not board.is_ended(state):
        # Choose a random legal action, do it, and repeat until the game is over
        chosenAction = choice(board.legal_actions(state))
        state = board.next_state(state, chosenAction)

    return state

# Done
def backpropagate(node: MCTSNode | None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    # The signature says that node can be None and the parent of a root node is probably none, so that should just be our exit condition
    while node is not None:
        # Update the visit count
        node.visits += 1

        # If the bot won, update the win count
        if won:
            node.wins += 1

        # Move to the parent node
        node = node.parent

        # Also, the slides said to do this recursively, but that doesn't really seem necessary when the loop is so simple (unless im really off here and in that case, whoops!)


# Done
def ucb(node: MCTSNode, is_opponent: bool):
    """ Calculates the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """

    # Equation was given in the lecture: (node wins) / (node visits) + (exploration factor) * sqrt(ln(parent node visits) / (node visits))

    explorationFactor = sqrt(2) # Wikipedia says that the exploration factor is usually sqrt(2) and I dont see any instuctions in the lecture or the assignment so.....

    winRate = node.wins / node.visits

    # Adjust winRate if the last action was performed by the opponent
    if is_opponent:
        winRate = 1 - winRate

    return winRate + (explorationFactor * sqrt(log(node.parent.visits) / node.visits))


# Done
def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """

    # Let's just loop through the child nodes and see which one has the highest win rate
    bestAction = None
    bestWinRate = -9999

    for action, child in root_node.child_nodes.items():
        winRate = child.wins / child.visits
        if winRate > bestWinRate:
            bestWinRate = winRate
            bestAction = action

    return bestAction


def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1


def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state)  # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    for _ in range(num_nodes):
        state = current_state
        node = root_node

        # Do MCTS - This is all you!
        # AAAA - I don't want it to be all me. It's scary

        #Step 1: Traverse the tree
        while not node.untried_actions and node.child_nodes:
            node, state = traverse_nodes(node, board, state, bot_identity)

        #Step 2: expand the leaf if we can
        if node.untried_actions:
            node, state = expand_leaf(node, board, state)

        #Step 3: Rollout to see how this might play out
        rolloutState = rollout(board, state)

        #Step 4: Backpropagate
        won = is_win(board, rolloutState, bot_identity)
        backpropagate(node, won)

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node)

    print(f"Action chosen: {best_action}")
    return best_action
