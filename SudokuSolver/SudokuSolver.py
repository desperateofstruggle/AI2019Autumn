import copy
import sys
import time
import os

import numpy as np

'''
class declaration: the class to solve the Sudoku by BF(brute force (exhaustive search) method )
'''

class SolutionBF:
    '''
    function declaration: the main interface function called by object instance to get if the problem be solved or not
    Parameters:
        board - the initial sudoku board
    Returns:
    '''

    def solves(self, board):
        self.sudokuboard = copy.deepcopy(board)
        self.answerboard = []
        self.issolved = False
        self.model = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.nodecount = 0
        self.times = 0
        self.time_start = time.time()
        self.time_end = 0

        self.fixed = copy.deepcopy(self.sudokuboard)
        for i in range(9):
            for j in range(9):
                if self.sudokuboard[i][j] != 0:
                    self.fixed[i][j] = 1

        self.solve(0, 0)
        self.time_end = time.time()
        self.times = self.time_end - self.time_start

    '''
    function declaration: to solve the sudokuboard (implement function)
    Parameters:
        x - the row of the box to handle
        y - the col of the box to handle
    Returns:
        True - there is at least one solution
        False - there is no solution
    '''

    def solve(self, x, y):

        if y == 9:
            if x == 8:
                self.answerboard.append(copy.deepcopy(self.sudokuboard))
                self.issolved = True
                # print(self.nodecount)
            else:
                self.solve(x + 1, 0)
        else:
            if self.fixed[x][y] == 1:
                self.solve(x, y + 1)
            else:
                for num in self.model:
                    self.nodecount = self.nodecount + 1
                    valid = self.isconsistent(x, y, num)

                    if valid:
                        self.sudokuboard[x][y] = num
                        self.solve(x, y + 1)
                self.sudokuboard[x][y] = 0
        return False

    '''
    function declaration: to get the empty (row, col) in the board
    Parameters:
    Returns:
        i - the empty index of row(-1 for none)
        j - the empty index of col(-1 for none)
    '''

    def findempty(self, x, y):
        for i in range(3):
            for j in range(3):
                if self.sudokuboard[x * 3 + i][y * 3 + j] == 0:
                    return x * 3 + i, y * 3 + j
        return -1, -1

    '''
    function declaration: interface to apply checking if it is consistent while adding the tmp into (row, col)
    Parameters:
        row - the row index in the board
        col - the col index in the board
        tmp - the label to check
    Returns:
        True - consistent
        False - inconsistent
    '''

    def isconsistent(self, row, col, tmp):
        if self.checkconsistence(row, col, tmp):
            return True
        return False


    '''
    function declaration: the implement of checking if it is consistent while adding the num into (row, col)
    Parameters:
        row - the row index in the board
        col - the col index in the board
        num - the label to check
    Returns:
        True - consistent
        False - inconsistent
    '''

    def checkconsistence(self, row, col, num):
        for i in range(9):
            if self.sudokuboard[row][i] == num:
                return False
        for j in range(9):
            if self.sudokuboard[j][col] == num:
                return False
        br, bc = self.getsmallboardstartxy(row, col)
        for r in range(br, br + 3):
            for c in range(bc, bc + 3):
                if self.sudokuboard[r][c] == num:
                    return False
        return True

    '''
    function declaration: get the small borad start x and y
    Parameters:
        row - the row index in the large board
        col - the col index in the large board
    Returns:
        brow - the start row index of the small board in the large board
        bcol - the start col index of the small board in the large board
    '''

    def getsmallboardstartxy(self, row, col):
        return row - row % 3, col - col % 3

    '''
    functions declaration: the api about some result information
    '''

    def getboard(self):
        return self.answerboard

    def getanswer(self):
        return self.issolved

    def getnodecount(self):
        return self.nodecount

    def gettime(self):
        return self.times

    def getstarttime(self):
        return self.time_start

    def getendtime(self):
        return self.time_end


'''
class declaration: the class to solve the Sudoku by BT(back-tracking (CSP))
'''

class SolutionBT:
    '''
    function declaration: the main interface function called by object instance to get if the problem be solved or not
    Parameters:
    Returns:
        True - there is at least one solution
        False - there is no solution
    '''

    def solves(self, board):
        self.sudokuboard = copy.deepcopy(board)
        self.model = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.answerboard = []
        self.issolved = False
        self.nodecount = 0
        self.times = 0
        self.time_start = time.time()
        self.time_end = 0
        if self.solve():
            self.answerboard.append(self.sudokuboard)
            self.issolved = True
            self.time_end = time.time()
            self.times = self.time_end - self.time_start
            return True
        self.time_end = time.time()
        self.times = self.time_end - self.time_start
        return False

    '''
    function declaration: to solve the sudokuboard (implement function)
    Parameters:
    Returns:
        True - there is at least one solution
        False - there is no solution
    '''

    def solve(self):
        row, col = self.findempty()
        if row == -1 or col == -1:
            return True
        for tmp in self.model:
            self.nodecount = self.nodecount + 1
            if self.isconsistent(row, col, tmp):
                self.sudokuboard[row][col] = tmp
                if self.solve():
                    return True
                self.sudokuboard[row][col] = 0
        return False

    '''
    function declaration: to get the empty (row, col) in the board
    Parameters:
    Returns:
        i - the empty index of row(-1 for none)
        j - the empty index of col(-1 for none)
    '''

    def findempty(self):
        for i in range(9):
            for j in range(9):
                if self.sudokuboard[i][j] == 0:
                    return i, j
        return -1, -1

    '''
    function declaration: interface to apply checking if it is consistent while adding the tmp into (row, col)
    Parameters:
        row - the row index in the board
        col - the col index in the board
        tmp - the label to check
    Returns:
        True - consistent
        False - inconsistent
    '''

    def isconsistent(self, row, col, tmp):
        if self.checkconsistence(row, col, tmp):
            return True
        return False

    '''
    function declaration: the implement of checking if it is consistent while adding the num into (row, col)
    Parameters:
        row - the row index in the board
        col - the col index in the board
        num - the label to check
    Returns:
        True - consistent
        False - inconsistent
    '''

    def checkconsistence(self, row, col, num):
        for i in range(9):
            if self.sudokuboard[row][i] == num:
                return False
        for j in range(9):
            if self.sudokuboard[j][col] == num:
                return False
        br, bc = self.getsmallboardstartxy(row, col)
        for r in range(br, br + 3):
            for c in range(bc, bc + 3):
                if self.sudokuboard[r][c] == num:
                    return False
        return True

    '''
    function declaration: get the small borad start x and y
    Parameters:
        row - the row index in the large board
        col - the col index in the large board
    Returns:
        brow - the start row index of the small board in the large board
        bcol - the start col index of the small board in the large board
    '''

    def getsmallboardstartxy(self, row, col):
        return row - row % 3, col - col % 3

    def getboard(self):
        return self.answerboard

    def getanswer(self):
        return self.issolved

    def getnodecount(self):
        return self.nodecount

    def gettime(self):
        return self.times

    def getstarttime(self):
        return self.time_start

    def getendtime(self):
        return self.time_end


'''
class declaration: the class to solve the Sudoku by FC-MRV(forward-checking with MRV heuristics )
'''

class SolutionFCMRV:

    '''
    function declaration: the main interface function called by object instance to get if the problem be solved or not
    Parameters:
        board - the initial Sudoku board to be solved
    Returns:
    '''

    def solves(self, board):
        self.sudokuboard = copy.deepcopy(board)
        self.candidate = [0, 0, 0, 0, 0, 0, 0, 0, 0]    # to record the candidate number(1 for can[i] means need, 0 means no need)
        self.path = []                                  # to record the route of the solutions
        self.CurDom = []                                # Optional field for each location, 0 for the first element of CurDom[i]
        for i in range(81):
            self.CurDom.append([0])
        self.Fixed = [0] * 81                           # the matrix to record the data
        self.answerboard = []
        self.issolved = False
        self.nodecount = 0
        self.times = 0
        self.time_start = time.time()
        self.time_end = 0

        self.solve()

        self.time_end = time.time()
        self.times = self.time_end - self.time_start

    '''
    function declaration: to solve the sudokuboard (implement function)
    Parameters:
    Returns:
    '''

    def solve(self):
        inits = 0
        for i in range(9):
            for j in range(9):
                index = i * 9 + j
                if self.sudokuboard[i][j] == 0:
                    inits = inits + 1
                    self.AttainCandidate(i, j)
                else:
                    self.Fixed[i * 9 + j] = 1
        self.issolved = self.iterate(inits)
        for p in self.path:
            self.sudokuboard[p[0]][p[1]] = p[2]
        self.answerboard.append(copy.deepcopy(self.sudokuboard))

    '''
    function declaration: to attain the reasonable candidate of (i, j)
    Parameters:
        i - the row index
        j - the col index
    Returns:
    '''

    def AttainCandidate(self, i, j):
        for tmp in range(9):
            self.candidate[tmp] = tmp + 1
        for col in range(9):
            if self.sudokuboard[i][col] != 0:
                self.candidate[self.sudokuboard[i][col] - 1] = 0
        for row in range(9):
            if self.sudokuboard[row][j] != 0:
                self.candidate[self.sudokuboard[row][j] - 1] = 0
        for row in range(int(i / 3) * 3, int(i / 3) * 3 + 3):
            for col in range(int(j / 3) * 3, int(j / 3) * 3 + 3):
                if self.sudokuboard[row][col] != 0:
                    self.candidate[self.sudokuboard[row][col] - 1] = 0

        for tmp in range(9):
            if self.candidate[tmp] != 0:
                self.CurDom[i * 9 + j].append(copy.deepcopy(self.candidate[tmp]))

    '''
    function declaration: to gain the worst number choice (means the best new points to consider)
    Parameters:
    Returns:
        results - the index in the CurDom
    '''

    def gainUnassingnedVal(self):
        mins = 10
        results = 80
        for i in range(81):
            if self.Fixed[i] == 0 and len(self.CurDom[i]) < mins:
                mins = len(self.CurDom[i])
                results = i
        return results

    '''
    function declaration: to call CurDomDelete to delete element D in the CurDom of the index
    Parameters:
        index - the index of the CurDom to be handle
        D     - the element to be deleted in the CurDom
    Returns:
        temp  - the index set of all the index to be deleted
    '''

    def delete_m(self, index, D):
        row = int(index / 9)
        col = index % 9
        srow = int(row / 3) * 3
        scol = int(col / 3) * 3
        temp = []
        for i in range(9):
            r_index = row * 9 + i
            if self.Fixed[r_index] == 0 and self.CurDomDelete(r_index, D):
                temp.append(copy.deepcopy(r_index))
            c_index = i * 9 + col
            if self.Fixed[c_index] == 0 and self.CurDomDelete(c_index, D):
                temp.append(copy.deepcopy(c_index))

        for i in range(srow, srow + 3):
            for j in range(scol, scol + 3):
                s_index = i * 9 + j
                if self.Fixed[s_index] == 0 and self.CurDomDelete(s_index, D):
                    temp.append(copy.deepcopy(s_index))
        return temp

    '''
    function declaration: to delete element D in the CurDom of the index
    Parameters:
    Returns:
        True  - successful
        False - unsuccessful
    '''

    def CurDomDelete(self, index, D):
        for i in range(1, len(self.CurDom[index])):
            if self.CurDom[index][i] == D:
                temp = []
                for j in range(1, len(self.CurDom[index])):
                    if self.CurDom[index][j] != D:
                        temp.append(copy.deepcopy(self.CurDom[index][j]))
                self.CurDom[index].clear()
                self.CurDom[index].append(0)
                for k in range(len(temp)):
                    self.CurDom[index].append(copy.deepcopy(temp[k]))
                return True
        return False

    '''
    function declaration: to judge if it can be continued
    Parameters:
    Returns:
        True  - ok
        False - no
    '''

    def Judge(self):
        for i in range(81):
            if self.Fixed[i] == 0 and len(self.CurDom[i]) == 1:
                return False
        return True

    '''
    function declaration: the function to recursion call
    Parameters:
    Returns:
        True  - ok
        False - no
    '''

    def iterate(self, height):
        if height == 0:
            return True
        best = self.gainUnassingnedVal()
        row = int(best / 9)
        col = best % 9
        for k in range(1, len(self.CurDom[best])):
            self.nodecount = self.nodecount + 1
            tmp = self.CurDom[best][k]
            self.Fixed[best] = tmp
            tmpList = self.delete_m(best, tmp)
            paths = []
            # save the path to fill with the board
            paths.append(copy.deepcopy(row))
            paths.append(copy.deepcopy(col))
            paths.append(copy.deepcopy(tmp))
            self.path.append(copy.deepcopy(paths))
            if self.Judge():
                if self.iterate(height - 1):
                    return True
            # status recovery
            for i in range(len(tmpList)):
                self.CurDom[tmpList[i]].append(copy.deepcopy(tmp))
            self.Fixed[best] = 0
            self.path = copy.deepcopy(self.path[0: len(self.path) - 1])
        return False

    def getboard(self):
        return self.answerboard

    def getanswer(self):
        return self.issolved

    def getnodecount(self):
        return self.nodecount

    def gettime(self):
        return self.times

    def getstarttime(self):
        return self.time_start

    def getendtime(self):
        return self.time_end


def inputFileHandler(address):
    file = open(address)
    file_lines = file.readlines()
    dataArray = np.zeros((9, 9))
    index = 0
    for line in file_lines:
        line = line.strip()
        formLine = line.split(' ')
        dataArray[index, :] = formLine[0:9]
        index = index + 1
    return dataArray.astype(int)

def outputFileHandler(address, matrix):
    with open(address, 'a+') as f:
        f.seek(0)
        f.truncate()
        for r in matrix:
            for e in r:
                f.write(str(e) + " ")
            f.write("\n")
        f.close()

def outputFilePerformanceHandler(address, strs):
    with open(address, 'a+') as f:
        f.seek(0)
        f.truncate()
        f.write(strs)
        f.close()

def printers():
    print("The number of the user-defined command arguments must be equal to 3 as follows:")
    print(">> python SudokuSolver.py [file path and file name] [BF/BT/FC-MRV]")
    print("Attention:other inputs will be considered to be the invalid input types!!!!")

def main():
    if len(sys.argv) != 3:
        printers()
        exit(-1)
    filesStr = sys.argv[1]
    programType = sys.argv[2]
    if programType != "BT" and programType != "BF" and programType != "FC-MRV":
        printers()
        exit(-1)

    cnt = 1
    t = time.time()

    res = inputFileHandler(filesStr)
    print("initial Sudoku board:")
    for r in res:
        for e in r:
            print(e,  end=" ")
        print()

    if programType == "BF":
        k = SolutionBF()
    elif programType == "BT":
        k = SolutionBT()
    elif programType == "FC-MRV":
        k = SolutionFCMRV()

    k.solves(res)

    if k.getanswer():
        t = k.getendtime() - t
        for r in k.getboard():
            if cnt == 1:
                outputFileHandler(filesStr.replace("puzzle", "solution"), r)
                print("file ", filesStr.replace("puzzle", "solution"), "writes successfully")
                tmp = ""
                tmp = tmp + "Total clock time: " + str(t) + "\n"
                tmp = tmp + "Search clock time: " + str(k.gettime()) + "\n"
                tmp = tmp + "Number of nodes generated: " + str(k.getnodecount()) + "\n"
                outputFilePerformanceHandler(filesStr.replace("puzzle", "performance" + str(programType)), tmp)
                print("file ", filesStr.replace("puzzle", "performance"), "writes successfully")
            print("result", cnt, ":")
            cnt = cnt + 1
            for c in r:
                for o in c:
                    print(o, end=' ')
                print("")
        print("nodecount:", k.getnodecount())
        print("searchtime:", k.gettime())
        print("alltime:", t)
    else:
        print("no solutions!")


if __name__ == '__main__':
    main()
