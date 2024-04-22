### Mancala Environment
### stores board state and implements the rules of the game

import numpy as np


class MancalaBoard:
    """An RL environment that knows the rules of mancala"""

    def __init__(self):
        self.reset()

    def reset(self, player: int = 0) -> np.array:
        """Reset the board to its original state."""
        self.board: np.array = np.array([4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0])
        self.turn: int = 0
        self.gameOver: bool = False

        if player == 1:
            action = np.random.randint(6)
            self.step_test(action, 0)
            if action == 2:
                action = np.random.choice(np.where(self.board[:6] > 0)[0])
                self.step_test(action, 0)
        return self.board

    def __str__(self):
        game = "\n________________\n"
        game += f"|      {self.board[13]}      |\n"
        for i in range(0, 6):
            game += f"|  {self.board[i]}   |   {self.board[12 - i]}  |\n"
        game += f"|      {self.board[6]}      |\n"
        game += "________________\n"
        return game

    def _checkGameOver(self):
        """Check if the game is over."""
        if np.sum(self.board[0:6]) == 0:
            for i in range(7, 13):
                self.board[13] += self.board[i]
                self.board[i] = 0
            self.gameOver = True
        elif np.sum(self.board[7:13]) == 0:
            for i in range(0, 6):
                self.board[6] += self.board[i]
                self.board[i] = 0
            self.gameOver = True

    def _move(self, action: int, goal_i: int, skip: int) -> int:
        # move all the pieces around the board
        pieces = self.board[action]
        dest_i = action
        self.board[action] = 0
        captured = 0
        for _ in range(pieces):
            dest_i = (dest_i + 1) % 14
            # don't put in your opponent's goal
            if dest_i == skip:
                dest_i = (dest_i + 1) % 14
            self.board[dest_i] += 1
        # if you landed on an empty square on your side, move all pieces to your goal
        if (
            self.board[dest_i] == 1
            and self.board[12 - dest_i] > 0
            and dest_i < goal_i
            and dest_i >= (skip + 1) % 14
        ):
            captured = self.board[12 - dest_i]
            self.board[goal_i] += self.board[12 - dest_i] + 1
            self.board[dest_i] = 0
            self.board[12 - dest_i] = 0

        ### game over cases ###
        self._checkGameOver()
        return dest_i, captured

    def step_test(self, action: int, player: int):
        """Determines effect of an action on the board for testing"""
        self.turn += 0 if player == 0 else 1
        goal_i: int = 6 if player == 0 else 13
        skip: int = 13 if player == 0 else 6

        # move
        dest_i, _ = self._move(action, goal_i, skip)

        # return the experience data
        return (
            self.board.copy(),
            player if dest_i == goal_i else (player + 1) % 2,
            self.gameOver,
        )

    def step(self, action: int, player: int):
        """Determines effect of an action on the board."""
        old_board = self.board.copy()
        self.turn += 1
        goal_i: int = 6 if player == 0 else 13
        skip: int = 13 if player == 0 else 6

        # move
        dest_i, captured = self._move(action, goal_i, skip)
        repeat = dest_i == goal_i

        ## opponent's moves
        if dest_i != goal_i:
            dest_i, goal_i, skip = skip, skip, goal_i
            while dest_i == goal_i and not self.gameOver:
                action = (
                    np.random.choice(np.where(self.board[goal_i - 6 : goal_i] > 0)[0])
                    + goal_i
                    - 6
                )
                dest_i, _ = self._move(action, goal_i, skip)

        # return the experience data
        return (
            self.board.copy(),
            self._getReward(old_board, repeat, captured, player),
            self.gameOver,
        )

    def _getReward(
        self,
        old_board: np.ndarray,
        bonus_move: bool = False,
        captured: int = 0,
        player: int = 0,
    ):
        """Caluclate reward based on turn and game results."""
        goal = 6 if player == 0 else 13
        their_goal = 13 if player == 0 else 6
        your_score: int = self.board[goal]
        their_score: int = self.board[their_goal]
        if self.gameOver:
            if your_score > their_score:
                return 2.0
            elif their_score > your_score:
                return -2.0
            else:
                return 0.0
        reward = 0.0
        if bonus_move:
            reward += 0.1
        reward += 0.1 * captured
        reward += 0.01 * (self.board[goal] - old_board[goal])
        reward -= 0.01 * (self.board[their_goal] - old_board[their_goal])
        return reward
