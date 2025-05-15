import numpy as np
import copy
import math
import cv2
import time
import scipy.signal as ss


class Node(object):

    def __init__(self, grid):

        """

        :param grid: np.array nrow*ncolumn

        author: weiwei
        date: 20190828, 20200104
        """

        self.grid = copy.deepcopy(grid)
        self._nrow, self._ncolumn = self.grid.shape
        self.ngrids = self._nrow * self._ncolumn
        self.parent = None
        self.gs = 0

    @property
    def nrow(self):
        return self._nrow

    @property
    def ncolumn(self):
        return self._ncolumn

    def __getitem__(self, x):
        return self.grid[x]

    def __eq__(self, anothernode):
        """
        determine if two nodes are the same
        :return:

        author: weiwei
        date: 20190828
        """

        return np.array_equal(self.grid, anothernode.grid)

    def __repr__(self):
        """
        overload the printed results

        :return:

        author: weiwei
        date: 20191003
        """

        outstring = "["
        for i in range(self._nrow):
            if i == 0:
                outstring += "["
            else:
                outstring += " ["
            for j in range(self._ncolumn):
                outstring = outstring + str(self.grid[i][j]) + ","
            outstring = outstring[:-1] + "]"
            outstring += ",\n"
        outstring = outstring[:-2] + "]]"

        return outstring


class TubePuzzle(object):

    def __init__(self, elearray):
        """

        :param nrow:
        :param ncolumn:
        :param elearray: nrow*ncolumn int array, tube id starts from 1, maximum 4

        author: weiwei
        date: 20191003
        """

        self._nrow = elearray.shape[0]
        self._ncolumn = elearray.shape[1]
        self.elearray = np.zeros((self._nrow, self._ncolumn), dtype="int")
        self.openlist = []
        self.closelist = []
        self._setValues(elearray)
        self.goalpattern = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [2, 2, 2, 2, 0, 0, 1, 1, 1, 1],
                                     [2, 2, 2, 2, 0, 0, 1, 1, 1, 1],
                                     [2, 2, 2, 2, 0, 0, 1, 1, 1, 1],
                                     [2, 2, 2, 0, 0, 0, 0, 1, 1, 1],
                                     [2, 2, 2, 0, 0, 0, 0, 1, 1, 1]])
        self.standpattern = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    def _setValues(self, elearray):
        """
        change the elements of the puzzle using state

        :param elearray: 2d array
        :return:

        author: weiwei
        date: 20190828, 20200104osaka
        """

        if elearray.shape != (self._nrow, self._ncolumn):
            print("Wrong number of elements in elelist!")
            raise Exception("Number of elements error!")
        self.elearray = elearray

    def _hs(self, node):
        """
        heuristics

        :return:

        author: weiwei
        date: 20200104
        """

        return np.sum((self.goalpattern != 1) * (node.grid == 1) + (self.goalpattern != 2) * (node.grid == 2))

    def isdone(self, node):
        """

        :return:

        author: weiwei
        date: 20190828
        """

        if np.any((self.goalpattern != 1) * (node.grid == 1)) or np.any((self.goalpattern != 2) * (node.grid == 2)):
            return False
        return True

    def fcost(self, node):
        hs = self._hs(node)
        gs = node.gs
        return hs + gs, hs, gs

    def getMovableFillablePair(self, node):
        """
        get a list of movable and fillable pairs

        :param node see Node
        :return: [[(i,j), (k,l)], ...]

        author: weiwei
        date: 20191003osaka, 20200104osaka
        """

        # filtering
        # mask_ulbr = np.array([[1,0,0],[0,0,0],[0,0,1]])
        # mask_urbl = np.array([[0,0,1],[0,0,0],[1,0,0]])
        # mask_ulbr2 = np.array([[1,0,0],[1,0,0],[0,1,1]])
        # mask_urbl2 = np.array([[0,1,1],[1,0,0],[1,0,0]])
        # mask_ulbr2_flp = np.array([[1,1,0],[0,0,1],[0,0,1]])
        # mask_urbl2_flp = np.array([[0,0,1],[0,0,1],[1,1,0]])
        mask_ucbc = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
        mask_crcl = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
        # mask_ul = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]])
        # mask_ur = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])
        # mask_bl = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])
        # mask_br = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
        ## fillable
        fillablegrid = np.zeros_like(node.grid)
        fillablegrid[self.standpattern==1] = node.grid[self.standpattern==1]
        cf_ucbc = ss.correlate2d(fillablegrid, mask_ucbc)[1:-1, 1:-1]
        cf_crcl = ss.correlate2d(fillablegrid, mask_crcl)[1:-1, 1:-1]
        cf = ((cf_ucbc==0)+(cf_crcl==0))*(node.grid==0)
        # fillable 1
        fillable_type1 = [np.asarray(np.where((self.goalpattern == 1) * cf)).T[0]]
        # fillable 2
        fillable_type2 = [np.asarray(np.where((self.goalpattern == 2) * cf)).T[0]]
        ## graspable
        fillablegrid = np.zeros_like(node.grid)
        fillablegrid[self.standpattern==0] = node.grid[self.standpattern==0]
        cm_ucbc = ss.correlate2d(fillablegrid, mask_ucbc)[1:-1, 1:-1]
        cm_crcl = ss.correlate2d(fillablegrid, mask_crcl)[1:-1, 1:-1]
        cm_ucbc[node.grid == 0] = -1
        cm_crcl[node.grid == 0] = -1
        # cm_ul[node.grid == 0] = -1
        # cm_ur[node.grid == 0] = -1
        # cm_bl[node.grid == 0] = -1
        # cm_br[node.grid == 0] = -1
        # cm = (cm_ulbr==0)+(cm_urbl==0)+(cm_ulbr_flp==0)+(cm_urbl_flp==0)+(cm_ucbc==0)+(cm_crcl==0)
        # cm = (cm_ulbr==0)+(cm_urbl==0)+(cm_ucbc==0)+(cm_crcl==0)
        cm = (cm_ucbc == 0) + (cm_crcl == 0)
        # cm = (cm_ucbc == 0) + (cm_crcl == 0) + (cm_ul == 0) + (cm_ur == 0) + (cm_bl == 0) + (cm_br == 0)
        # movable 1
        movable_type1 = np.asarray(np.where(cm * (node.grid == 1) * (self.standpattern == 0))).T
        # movable 2
        movable_type2 = np.asarray(np.where(cm * (node.grid == 2) * (self.standpattern == 0))).T
        movable_expanded_type1 = np.repeat(movable_type1, len(fillable_type1), axis=0)
        movable_expanded_type2 = np.repeat(movable_type2, len(fillable_type2), axis=0)
        if len(movable_expanded_type1) == 0:
            movableeles = movable_expanded_type2
        elif len(movable_expanded_type2) == 0:
            movableeles = movable_expanded_type1
        else:
            movableeles = np.concatenate((movable_expanded_type1, movable_expanded_type2), axis=0)
        fillable_expanded_type1 = np.tile(fillable_type1, (len(movable_type1), 1))
        fillable_expanded_type2 = np.tile(fillable_type2, (len(movable_type2), 1))
        if len(fillable_expanded_type1) == 0:
            fillableeles = fillable_expanded_type2
        elif len(fillable_expanded_type2) == 0:
            fillableeles = fillable_expanded_type1
        else:
            fillableeles = np.concatenate((fillable_expanded_type1, fillable_expanded_type2), axis=0)
        return movableeles, fillableeles

    def _reorderopenlist(self):
        self.openlist.sort(key=lambda x: (self.fcost(x)[0], self.fcost(x)[1]))

    def atarSearch(self, weightarray=None):
        """

        build a graph considering the movable and fillable ids

        :param weightarray

        :return:

        author: weiwei
        date: 20191003
        """

        if weightarray is None:
            weightarray = np.zeros_like(self.elearray)

        startnode = Node(self.elearray)
        self.openlist = [startnode]
        while True:
            # if len(self.openlist)>=2:
            #     for eachnode in self.openlist:
            #         print(eachnode)
            #         print(eachnode.fcost())
            #     print("\n")
            self._reorderopenlist()
            print(self.openlist[0])
            print(self.fcost(self.openlist[0]))
            print("\n")
            self.closelist.append(self.openlist.pop(0))
            # movableids = self.getMovableIds(self.closelist[-1])
            # fillableids = self.getFillableIds(self.closelist[-1])
            # if len(movableids) == 0 or len(fillableids) == 0:
            #     print("No path found!")
            #     return []
            # for mid in movableids:
            #     for fid in fillableids:
            movableeles, fillableeles = self.getMovableFillablePair(self.closelist[-1])
            if movableeles.shape[0] == 0:
                print("No path found!")
                return []
            for i in range(movableeles.shape[0]):
                mi, mj = movableeles[i]
                fi, fj = fillableeles[i]
                # only consider goal here, since the goal will not be changed
                if weightarray[mi, mj] != 0 and weightarray[fi, fj] != 0:
                    continue
                tmpelearray = copy.deepcopy(self.closelist[-1])
                tmpelearray.parent = self.closelist[-1]
                tmpelearray.gs = self.closelist[-1].gs + 1
                tmpelearray[fi][fj] = tmpelearray[mi][mj]
                tmpelearray[mi][mj] = 0
                #  check if path is found
                if self.isdone(tmpelearray):
                    path = [tmpelearray]
                    parent = tmpelearray.parent
                    while parent is not None:
                        path.append(parent)
                        parent = parent.parent
                    print("Path found!")
                    print(tmpelearray)
                    print(self.fcost(tmpelearray))
                    # for eachnode in path:
                    #     print(eachnode)
                    return path[::-1]
                # check if in openlist
                flaginopenlist = False
                for eachnode in self.openlist:
                    if eachnode == tmpelearray:
                        flaginopenlist = True
                        if self.fcost(eachnode)[0] <= self.fcost(tmpelearray)[0]:
                            pass
                            # no need to update position
                        else:
                            eachnode.parent = tmpelearray.parent
                            # self._reorderopenlist()
                        # continue
                        break
                if flaginopenlist:
                    continue
                else:
                    # not in openlist append and sort openlist
                    self.openlist.append(tmpelearray)


if __name__ == "__main__":
    # down x, right y
    elearray = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                         [1, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                      [2, 2, 0, 2, 1, 0, 0, 0, 0, 0],
    #                      [1, 1, 0, 1, 2, 0, 0, 0, 0, 2],
    #                      [0, 2, 0, 0, 0, 0, 0, 0, 0, 2]])
    # state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                      [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
    #                      [0, 0, 2, 1, 1, 1, 0, 0, 0, 0],
    #                      [0, 0, 2, 1, 2, 2, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 2, 0, 0, 0, 0, 0]])
    tp = TubePuzzle(elearray)
    # tp.getMovableIds(Node(state))
    # print(Node(state).fcost())
    # print(tp.fcost(Node(state)))
    path = tp.atarSearch()
    for node in path:
        print(node)
