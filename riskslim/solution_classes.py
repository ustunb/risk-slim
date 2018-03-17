import numpy as np
import prettytable as pt

class SolutionPool(object):
    """

    """
    def __init__(self,  obj):

        if isinstance(obj, SolutionPool):

            self._P = obj.P
            self._objvals = obj.objvals
            self._solutions = obj.solutions

        elif isinstance(obj, int):

            assert obj >= 1
            self._P = int(obj)
            self._objvals = np.empty(0)
            self._solutions = np.empty(shape = (0, self._P))

        elif isinstance(obj, dict):

            assert len(obj) == 2
            objvals = np.copy(obj['objvals']).flatten().astype(dtype = np.float_)
            solutions = np.copy(obj['solutions'])
            n = objvals.size
            if solutions.ndim == 2:
                assert n in solutions.shape
                if solutions.shape[1] == n and solutions.shape[0] != n:
                    solutions = np.transpose(solutions)
            elif solutions.ndim == 1:
                assert n == 1
                solutions = np.reshape(solutions, (1, solutions.size))
            else:
                raise ValueError('solutions has more than 2 dimensions')

            self._P = solutions.shape[1]
            self._objvals = objvals
            self._solutions = solutions

        else:
            raise ValueError('cannot initialize SolutionPool using %s object' % type(obj))


    def __len__(self):
        return len(self._objvals)


    def copy(self):
        return SolutionPool(self)


    @property
    def P(self):
        return int(self._P)


    @property
    def objvals(self):
        return np.copy(self._objvals)


    @property
    def solutions(self):
        return np.copy(self._solutions)


    @objvals.setter
    def objvals(self, objvals):
        if hasattr(objvals, "__len__"):
            if len(objvals) > 0:
                self._objvals = np.copy(list(objvals)).flatten().astype(dtype = np.float_)
            elif len(objvals) == 0:
                self._objvals = np.empty(0)
        else:
            self._objvals = float(objvals)


    @solutions.setter
    def solutions(self, solutions):
        if solutions.ndim == 2:
            assert self._P in solutions.shape
            if solutions.shape[0] == self._P and solutions.shape[1] != self._P:
                solutions = np.transpose(solutions)
        elif solutions.ndim == 1:
            solutions = np.reshape(solutions, (1, solutions.size))
        else:
            raise ValueError('incorrect solution dimensions')

        self._solutions = np.copy(solutions)


    def _generate(self, objvals, solutions):
        return SolutionPool({'objvals': objvals, 'solutions': solutions})


    def append(self, pool):
        return self.add(pool.objvals, pool.solutions)


    def add(self, objvals, solutions):

        if isinstance(objvals, np.ndarray) or isinstance(objvals, list):
            n = len(objvals)
            if solutions.ndim == 2:
                assert n in solutions.shape
                assert self._P in solutions.shape
                if solutions.shape[0] == self._P and solutions.shape[1] != self._P:
                    solutions = np.transpose(solutions)
            elif solutions.ndim == 1:
                assert n == 1
                solutions = np.reshape(solutions, (1, solutions.size))
            else:
                raise ValueError('incorrect solution dimensions')
        else:
            objvals = float(objvals) #also assertion
            solutions = np.reshape(solutions, (1, self._P))

        new = SolutionPool(self._P)
        new.objvals = np.append(self._objvals, objvals)
        new.solutions = np.append(self._solutions, solutions, axis = 0)
        return new


    def sort(self):
        idx = np.argsort(self._objvals)
        return SolutionPool({'objvals': self._objvals[idx], 'solutions': self._solutions[idx, :]})


    def distinct(self):
        _, idx = np.unique(self._solutions, return_index = True, axis = 0)
        return SolutionPool({'objvals': self._objvals[idx], 'solutions': self._solutions[idx, :]})


    def filter(self, filter_ind):
        idx = np.require(filter_ind, dtype = 'bool').flatten()
        return SolutionPool({'objvals': self._objvals[idx], 'solutions': self._solutions[idx, :]})


    def map(self, mapfun, target = 'all'):
        assert callable(mapfun), 'map function must be callable'
        if target is 'solutions':
            return map(mapfun, self.solutions)
        elif target is 'objvals':
            return map(mapfun, self.objvals)
        elif target is 'all':
            return map(mapfun, self.objvals, self.solutions)
        else:
            raise ValueError('target must be either solutions, objvals, or all')


    @staticmethod
    def isIntegral(solution):
        return np.all(solution == np.require(solution, dtype = 'int_'))


    def computeObjvals(self, getObjval):
        objvals = self._objvals
        compute_ind = np.flatnonzero(np.isnan(objvals))
        objvals[compute_ind] = map(getObjval, self._solutions[compute_ind])
        return self._generate(objvals, self._solutions)


    def removeSuboptimal(self, objval_cutoff):
        return self.filter(self.objvals <= objval_cutoff)


    def removeInfeasible(self, isFeasible):
        return self.filter(map(isFeasible, self.solutions))


    def removeNonintegral(self):
        return self.filter(map(self.isIntegral, self.solutions))


    def removeIntegral(self):
        return self.filter(~map(self.isIntegral, self.solutions))


    @staticmethod
    def solutionString(solution):
        solution_string = ''
        for j in range(len(solution)):
            if SolutionPool.isIntegral(solution[j]):
                solution_string += ' ' + str(int(solution[j]))
            else:
                solution_string += (' %1.4f' % solution[j])
        return solution_string


    def table(self):
        x = pt.PrettyTable(align = 'r', float_format = '1.4', hrules=pt.ALL)
        x.add_column("objval", self._objvals.tolist())
        x.add_column("solution", map(self.solutionString, self._solutions))
        return str(x)


    def __repr__(self):
        return self.table()



class SolutionQueue(object):
    """
    SolutionQueue is written to work faster than SolutionPool and is only used by the callback functions in risk_slim
    helper class used to create/manipulate a queue of solutions and objective values
    """

    def __init__(self, P):
        self._P = int(P)
        self._objvals = np.empty(shape = 0)
        self._solutions = np.empty(shape = (0, P))

    def __len__(self):
        return len(self._objvals)

    @property
    def P(self):
        return int(self._P)

    @property
    def objvals(self):
        return np.copy(self._objvals)

    @property
    def solutions(self):
        return np.copy(self._solutions)


    def add(self, new_objvals, new_solutions):
        if isinstance(new_objvals, np.ndarray) or isinstance(new_objvals, list):
            self._objvals = np.append(self._objvals, list(new_objvals))
            n = len(new_objvals)
        else:
            self._objvals = np.append(self._objvals, float(new_objvals))
            n = 1
        new_solutions = np.reshape(new_solutions, (n, self._P))
        self._solutions = np.append(self._solutions, new_solutions, axis = 0)

    def get_best_objval_and_solution(self):
        idx = np.argmin(self._objvals)
        return float(self._objvals[idx]), np.copy(self._solutions[idx,])


    def filter_sort_unique(self, max_objval = float('inf')):
        # filter
        if max_objval < float('inf'):
            good_idx = self._objvals <= max_objval
            self._objvals = self._objvals[good_idx]
            self._solutions = self._solutions[good_idx,]

        if len(self._objvals) > 0:
            _, unique_idx = np.unique(self._solutions, axis = 0, return_index = True)
            self._objvals = self._objvals[unique_idx]
            self._solutions = self._solutions[unique_idx,]

            sort_idx = np.argsort(self._objvals)
            self._objvals = self._objvals[sort_idx]
            self._solutions = self._solutions[sort_idx,]

    def clear(self):
        self._objvals = np.empty(shape = 0)
        self._solutions = np.empty(shape = (0, self._P))




