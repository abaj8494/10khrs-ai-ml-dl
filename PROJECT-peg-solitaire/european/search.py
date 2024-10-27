from collections import deque
from copy import deepcopy

EMPTY = -1
PEG = 1
VOID = 0
N = 7

class PegSolitaire:
    def __init__(self, board, moves):
        self.board = board
        self.moves = moves

    def is_solved(self):
        pegs = 0
        position = []
        for i, row in enumerate(board):
            for col in range(len(row)):
                if (board[i][col] > 0):
                    position.append([i, col])
                    pegs+= 1
            if (pegs > 1):
                return False
        #if (pegs == 1 and position == [[3,3]]):
        if (pegs == 1):
            return True


    def get_valid_moves(self):
        valids = []
        for i, row in enumerate(board):
            for col in range(len(row)):
                if (board[i][col] > 0):
                    if (i > 1 and board[i-2][col] < 0 and board[i-1][col] > 0):
                        valids.append([i,col,"w",board[i-1][col]])
                    if (col > 1 and board[i][col-2] < 0 and board[i][col-1] > 0): 
                        valids.append([i,col,"a",board[i][col-1]])
                    if (i < N-2 and board[i+2][col] < 0 and board[i+1][col] > 0):
                        valids.append([i,col,"s",board[i+1][col]])
                    if (col < N-2 and board[i][col+2] < 0 and board[i][col+1] > 0):
                        valids.append([i,col,"d",board[i][col+1]])
        #print("VALID MOVES:", valids)
        #print("MOVES thus far:", self.moves)
        return valids

    def make_move(self, move):
        self.moves.append(move)
        match move[2]:
            case "w":
                board[move[0]-1][move[1]] = -1
                board[move[0]-2][move[1]] = board[move[0]][move[1]]
                board[move[0]][move[1]] = -1
                move[0] -= 2
            case "a":
                board[move[0]][move[1]-1] = -1
                board[move[0]][move[1]-2] = board[move[0]][move[1]]
                board[move[0]][move[1]] = -1
                move[1] -= 2
            case "s":
                board[move[0]+1][move[1]] = -1
                board[move[0]+2][move[1]] = board[move[0]][move[1]]
                board[move[0]][move[1]] = -1
                move[0] += 2
            case "d":
                board[move[0]][move[1]+1] = -1
                board[move[0]][move[1]+2] = board[move[0]][move[1]]
                board[move[0]][move[1]] = -1
                move[1] += 2
    
    def undo_move(self):
        move = self.moves.pop()
        print("POPPED:", move)
        match move[2]:
            case "w":
                board[move[0]+2][move[1]] = board[move[0]][move[1]]
                board[move[0]][move[1]] = -1
                board[move[0]+1][move[1]] = move[3]
            case "a":
                board[move[0]][move[1]+2] = board[move[0]][move[1]]
                board[move[0]][move[1]] = -1
                board[move[0]][move[1]+1] = move[3]
            case "s":
                board[move[0]-2][move[1]] = board[move[0]][move[1]]
                board[move[0]][move[1]] = -1
                board[move[0]-1][move[1]] = move[3]
            case "d":
                board[move[0]][move[1]-2] = board[move[0]][move[1]] 
                board[move[0]][move[1]] = -1
                board[move[0]][move[1]-1] = move[3]

    def dfs(self):
        if self.is_solved():
            return True
        for move in self.get_valid_moves():
            self.print_board()
            self.make_move(move)
            if self.dfs():
                return True
            self.undo_move()
        return False

    def bfs(self):
        queue = deque()
        dc = deepcopy(self)
        queue.append(dc)
        visited = set()
        visited.add(str(dc))

        while queue:
            #self.print_board()
            curr_board = queue.popleft()
            #self.print_board()
            #self.print_moves()
            

            if curr_board.is_solved():
                print('im solved')
                self.moves = curr_board.moves
                return self.moves 
            for move in curr_board.get_valid_moves():
                neighbour_board = deepcopy(curr_board)
                neighbour_board.make_move(move)

                board_str = str(neighbour_board)
                if board_str not in visited:
                    visited.add(board_str)
                    queue.append(neighbour_board)

        print('hello')
        self.print_board()
        return None 

    def print_moves(self):
        print(self.moves)
        print(len(self.moves))
            
    def print_board(self):
        for row in board:
            for col in range(len(row)):
                if (row[col] > 0):
                    print('.', end='')
                elif (row[col] == 0):
                    print(' ', end='')
                elif (row[col] < 0):
                    print('O', end='')
            print('\n')

if __name__ == "__main__":
    board = [[ 0,  0,  1,  2,  3,  0,  0],
             [ 0,  4,  5,  6,  7,  8,  0],
             [ 9, 10, 11, 12, 13, 14, 15],
             [16, 17, 18, 36, 19, 20, 21],
             [22, 23, 24, 25, 26, 27, 28],
             [ 0, 29, 30, 31, 32, 33,  0],
             [ 0,  0, 34, 35, -1,  0,  0]]
    game = PegSolitaire(board, [])
    game.bfs()
    game.print_moves()

