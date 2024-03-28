### Mancala Environment
### stores board state and implements the rules of the game

import numpy as np


class MancalaBoard:
    """An RL environment that knows the rules of mancala"""

    def __init__(self):
        self.reset()

    def reset(self) -> np.array:
        """Reset the board to its original state."""
        self.board: np.array = np.array([4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0])
        self.turn: int = 0
        self.gameOver: bool = False
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

    def _move(
        self, action: int, pieces: int, goal_i: int, skip: int, player: int
    ) -> int:
        # move all the pieces around the board
        self.board[action] = 0
        dest_i = (action) % 14
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
            and (
                (dest_i < 6 and player == 0)
                or (dest_i >= 7 and dest_i < 13 and player == 1)
            )
        ):
            self.board[goal_i] += self.board[dest_i] + self.board[12 - dest_i]
            self.board[dest_i] = 0
            self.board[12 - dest_i] = 0

        ### game over cases ###
        self._checkGameOver()
        return dest_i

    def step(self, action: int):
        """Determines effect of an action on the board."""
        old_board = self.board.copy()
        self.turn += 1
        pieces: int = self.board[action]
        goal_i: int = 6
        skip: int = 13

        # move
        dest_i = self._move(action, pieces, goal_i, skip, 0)

        ## opponent's moves
        if dest_i != goal_i:
            dest_i, goal_i, skip = 13, 13, 6
            while dest_i == goal_i and not self.gameOver:
                action = np.random.choice(np.where(self.board[7:13] > 0)[0]) + 7
                dest_i = self._move(action, self.board[action], goal_i, skip, 1)

        # return the experience data
        return (
            self.board.copy(),
            self._getReward(old_board, dest_i == goal_i),
            self.gameOver,
        )

    def _getReward(self, old_board: np.ndarray, repeat: bool = False):
        """Caluclate reward based on turn and game results."""
        your_score: int = self.board[6]
        their_score: int = self.board[13]
        if self.gameOver:
            return 0.1 if your_score > their_score else -0.1
        else:
            return (
                (your_score - old_board[6] + 2 if repeat else 0)
                - (their_score - old_board[13])
            ) / 100.0
