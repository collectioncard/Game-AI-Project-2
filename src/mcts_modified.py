from mcts_node import MCTSNode
from p2_t3 import Board
from math import sqrt, log

num_nodes = 1000
explore_faction = 2.


def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
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
    while node.untried_actions == [] and node.child_nodes != {}:
        currentBestUcb = -9999
        bestChild = None

        for action, child in node.child_nodes.items():
            ucbValue = ucb(child, bot_identity == board.current_player(state))

            if ucbValue > currentBestUcb:
                currentBestUcb = ucbValue
                bestChild = child

        if bestChild is not None:
            node = bestChild
            state = board.next_state(state, bestChild.parent_action)

    return node, state


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
    if not node.untried_actions:
        return None, state

    actionToExpand = node.untried_actions.pop()
    newGameState = board.next_state(state, actionToExpand)
    newChildNode = MCTSNode(parent=node, parent_action=actionToExpand, action_list=board.legal_actions(newGameState))
    node.child_nodes[actionToExpand] = newChildNode

    return newChildNode, newGameState


def rollout(board: Board, state):
    """ Simulates possible game outcomes from the given state and returns the best final state it found

        Args:
            board:  The game setup.
            state:  The state of the game.

        Returns:
            state: The best terminal game state

        """
    current_player = board.current_player(state)
    opponent = 1 if current_player == 2 else 2

    while not board.is_ended(state):
        legal_actions = board.legal_actions(state)
        best_score = -9999999999999999999999
        best_action = None

        for action in legal_actions:
            new_state = board.next_state(state, action)
            score = heuristic(board, new_state, opponent, current_player, action)
            if score > best_score:
                best_score = score
                best_action = action

        state = board.next_state(state, best_action)

    return state


def heuristic(board: Board, state, opponent: int, currentPlayer: int, action):
    """ Evaluates the given state and returns a score for the bot

    Args:
        opponent: The id of the opponent player.
        action: The action to test
        currentPlayer: the id of the current player
        board:  The game setup.
        state:  The state of the game.

    Returns:
        score: The score of the state for the bot

    """
    pointGreatness = 0

    # Points if we are playing
    if currentPlayer is not opponent:
        # Win, duh
        if board.win_values(state) is not None:
            pointGreatness += 100000000000

        # We like center squares
        if action[2:] == (1, 1):
            pointGreatness += 100000000

        #And the corners
        if action[2:] in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            pointGreatness += 500

        # Finally avoid the side centers
        if action[2:] in [(1, 0), (0, 1), (1, 2), (2, 1)]:
            pointGreatness += 100
    else:
        # Points if we are not playing
        # Win, duh
        if board.win_values(state):
            pointGreatness -= 10000

        # We like center squares
        if action[2:] == (1, 1):
            pointGreatness -= 1000

        # And the corners
        if action[2:] in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            pointGreatness -= 500

        # Finally avoid the side centers
        if action[2:] in [(1, 0), (0, 1), (1, 2), (2, 1)]:
            pointGreatness -= 100

    return pointGreatness


def winning_move(board, state, player):
    for action in board.legal_actions(state):
        next_state = board.next_state(state, action)
        points_values = board.points_values(next_state)
        if points_values is not None and points_values.get(player, 0) == 1:
            return True
    return False


def backpropagate(node: MCTSNode | None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    while node is not None:
        node.visits += 1
        if won:
            node.wins += 1
        node = node.parent


def ucb(node: MCTSNode, is_opponent: bool):
    """ Calcualtes the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """
    explorationFactor = sqrt(2)
    winRate = node.wins / node.visits

    if is_opponent:
        winRate = 1 - winRate

    return winRate + (explorationFactor * sqrt(log(node.parent.visits) / node.visits))


def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
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
        # ...
        while not node.untried_actions and node.child_nodes:
            node, state = traverse_nodes(node, board, state, bot_identity)

        if node.untried_actions:
            node, state = expand_leaf(node, board, state)

        rolloutState = rollout(board, state)

        won = is_win(board, rolloutState, bot_identity)
        backpropagate(node, won)

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node)

    print(f"Action chosen: {best_action}")
    return best_action
