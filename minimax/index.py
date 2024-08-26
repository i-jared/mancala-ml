import math
import random


def minimax(board, depth, alpha, beta, maximizing_player, ai_first):
    if depth == 0 or game_over(board):
        return evaluate(board, ai_first)

    if maximizing_player:
        max_eval = -math.inf
        for move in get_valid_moves(board, ai_first):
            new_board, extra_turn = make_move(board, move, ai_first)
            eval = minimax(new_board, depth - 1, alpha,
                           beta, extra_turn, ai_first)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in get_valid_moves(board, not ai_first):
            new_board, extra_turn = make_move(board, move, not ai_first)
            eval = minimax(new_board, depth - 1, alpha,
                           beta, not extra_turn, ai_first)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def get_best_move(board, depth, ai_first, ai_turn=True):
    best_move = None
    best_eval = -math.inf if ai_turn else math.inf
    alpha = -math.inf
    beta = math.inf
    for move in get_valid_moves(board, ai_first):
        new_board, extra_turn = make_move(board, move, ai_first)
        eval = minimax(new_board, depth - 1, alpha, beta, extra_turn, ai_first)
        if ai_turn and eval > best_eval:
            best_eval = eval
            best_move = move
        elif not ai_turn and eval < best_eval:
            best_eval = eval
            best_move = move
        if ai_turn:
            alpha = max(alpha, eval)
        else:
            beta = min(beta, eval)
    return best_move


def game_over(board):
    return sum(board[:6]) == 0 or sum(board[7:13]) == 0


def evaluate(board, ai_first):
    ai_score = board[6] if ai_first else board[13]
    human_score = board[13] if ai_first else board[6]
    return ai_score - human_score


def get_valid_moves(board, is_player_one):
    start = 0 if is_player_one else 7
    return [i for i in range(start, start + 6) if board[i] > 0]


def make_move(board, move, player_turn):
    new_board = board.copy()
    start = 0 if player_turn else 7
    goal = 6 if player_turn else 13
    pieces = new_board[move]
    new_board[move] = 0
    current = move + 1

    while pieces > 0:
        if current == (13 if player_turn else 6):
            current = (0 if player_turn else 7)
            continue
        new_board[current] += 1
        pieces -= 1
        if pieces == 0:
            if current == goal:
                return new_board, True  # Extra turn
            if new_board[current] == 1 and current in range(start, start + 6):
                opposite = 12 - current
                if new_board[opposite] > 0:
                    new_board[goal] += new_board[opposite] + 1
                    new_board[opposite] = 0
                    new_board[current] = 0
        current = (current + 1) % 14

    return new_board, False


def play_mancala_against_human(depth):
    board = [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]
    # ai_first = random.choice([True, False])
    ai_first = False
    player_turn = not ai_first

    print("AI goes first" if ai_first else "You go first")

    while not game_over(board):
        print_board(board)

        if player_turn:
            while True:
                try:
                    move = int(
                        input("Enter your move (0-5 if you're first, 7-12 if you're second): "))
                    if move in get_valid_moves(board, not ai_first):
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            # AI uses minimax with depth 5
            move = get_best_move(board, depth, ai_first)
            print(f"AI's move: {move}")

        board, extra_turn = make_move(
            board, move, not ai_first if player_turn else ai_first)
        if not extra_turn:
            player_turn = not player_turn

    # End game: move all remaining pieces to respective goals
    for i in range(6):
        board[6] += board[i]
        board[i] = 0
    for i in range(7, 13):
        board[13] += board[i]
        board[i] = 0

    print_board(board)
    ai_score = board[6] if ai_first else board[13]
    human_score = board[13] if ai_first else board[6]
    if human_score > ai_score:
        print("You win!")
    elif ai_score > human_score:
        print("AI wins!")
    else:
        print("It's a tie!")


def print_board(board):
    print("  ", end="")
    for i in range(5, -1, -1):
        print(f"{board[i]:2d}", end=" ")
    print()
    print(f"{board[6]:2d}" + " " * 17 + f"{board[13]:2d}")
    print("  ", end="")
    for i in range(7, 13):
        print(f"{board[i]:2d}", end=" ")
    print("\n")


def play_against_random(n_games, depth):
    ai_wins = 0
    random_wins = 0
    ties = 0

    for n in range(n_games):
        board = [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]
        # ai_first = random.choice([True, False])
        ai_first = True
        player_turn = not ai_first

        while not game_over(board):
            if player_turn:
                valid_moves = get_valid_moves(board, not ai_first)
                move = random.choice(valid_moves)
            else:
                move = get_best_move(board, depth, ai_first)

            board, extra_turn = make_move(
                board, move, not ai_first if player_turn else ai_first)
            if not extra_turn:
                player_turn = not player_turn

        # End game: move remaining pieces to goals
        for i in range(6):
            board[6] += board[i]
            board[i] = 0
        for i in range(7, 13):
            board[13] += board[i]
            board[i] = 0

        ai_score = board[6] if ai_first else board[13]
        random_score = board[13] if ai_first else board[6]

        if random_score > ai_score:
            random_wins += 1
        elif ai_score > random_score:
            ai_wins += 1
        else:
            ties += 1
        # print(f"ai:{ai_score} random:{random_score}")

    print(f"Results after {n_games} games:")
    print(f"AI wins: {ai_wins}")
    print(f"Random wins: {random_wins}")
    print(f"Ties: {ties}")
    print(f"AI win rate: {ai_wins / n_games:.2%}")


def play_against_ai(n_games, depth1, depth2):
    bot1_wins = 0
    bot2_wins = 0
    ties = 0

    for n in range(n_games):
        board = [3, 2, 2, 0, 0, 1, 16, 2, 3, 1, 1, 2, 1, 14]
        # ai_first = random.choice([True, False])
        ai_first = False
        player_turn = False

        while not game_over(board):
            print_board(board)
            if player_turn:
                move = get_best_move(board, depth2, not ai_first)
                print(f"bot2's move: {move}")
            else:
                move = get_best_move(board, depth1, ai_first)
                print(f"bot1's move: {move}")

            board, extra_turn = make_move(
                board, move, not ai_first if player_turn else ai_first)
            if not extra_turn:
                player_turn = not player_turn

        # End game: move remaining pieces to goals
        for i in range(6):
            board[6] += board[i]
            board[i] = 0
        for i in range(7, 13):
            board[13] += board[i]
            board[i] = 0

        bot1_score = board[6] if ai_first else board[13]
        bot2_score = board[13] if ai_first else board[6]

        if bot2_score > bot1_score:
            bot2_wins += 1
        elif bot1_score > bot2_score:
            bot1_wins += 1
        else:
            ties += 1
        print(f"bot1:{bot1_score} bot2:{bot2_score}")

    print(f"Results after {n_games} games:")
    print(f"Bot1 wins: {bot1_wins}")
    print(f"Bot2 wins: {bot2_wins}")
    print(f"Ties: {ties}")
    print(f"AI win rate: {bot1_wins / n_games:.2%}")


# Uncomment the line below to run the function
play_against_ai(1, 15, 15)
# play_mancala_against_human(15)
