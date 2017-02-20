import numpy as np

class SolutionPool(object):
    """
    helper class used to create/manipulate a pool of solutions and objective values
    """

    @staticmethod
    def isIntegral(solution):
        return np.all(solution == np.require(solution, dtype = 'int_'))

    @staticmethod
    def solutionString(solution):
        solution_string = ''
        for j in range(0, len(solution)):
            if SolutionPool.isIntegral(solution[j]):
                solution_string += ' ' + str(int(solution[j]))
            else:
                solution_string += (' %1.4f' % solution[j])
        return solution_string

    def __init__(self, obj):
        obj_type = type(obj).__name__
        if obj_type == 'int':
            self.P = obj
            self.objvals = np.empty(shape = 0)
            self.solutions = np.empty(shape = (0, self.P))
        elif obj_type == 'SolutionPool':
            self.P = obj.P
            self.objvals = obj.objvals.copy()
            self.solutions = obj.solutions.copy()
        elif obj_type == 'dict':
            self.objvals = np.array(obj['objvals']).copy()
            self.solutions = np.array(obj['solutions']).copy()
            self.P = self.solutions.shape[1]
        else:
            raise ValueError('cannot intialize empty pool')

    def __len__(self):
        return len(self.objvals)

    def copy(self):
        new = SolutionPool(self.P)
        new.objvals = self.objvals.copy()
        new.solutions = self.solutions.copy()
        return new


    def sort(self):
        sort_ind = np.argsort(self.objvals)
        new = self.copy()
        new.objvals = new.objvals[sort_ind]
        new.solutions = new.solutions[sort_ind]
        return new

    def map(self, mapfun, target = 'all'):
        if target is 'solutions':
            return map(mapfun, self.solutions)
        elif target is 'objvals':
            return map(mapfun, self.objvals)
        elif target is 'all':
            return map(mapfun, self.objvals, self.solutions)
        else:
            raise ValueError('need target')

    def filter(self, filter_ind):
        filter_ind = np.require(filter_ind, dtype = 'bool')
        new = self.copy()
        new.objvals = self.objvals[filter_ind]
        new.solutions = self.solutions[filter_ind]
        return new

    def distinct(self):
        b = np.ascontiguousarray(self.solutions).view(np.dtype((np.void, self.solutions.dtype.itemsize * self.P)))
        _, unique_ind = np.unique(b, return_index = True)
        unique_ind = np.sort(unique_ind)
        new = self.copy()
        new.objvals = self.objvals[unique_ind]
        new.solutions = self.solutions[unique_ind]
        return new

    def add(self, objvals, solutions):
        if solutions.ndim == 1:
            solutions = np.reshape(solutions, (1, self.P))
        elif solutions.ndim == 2 and solutions.shape[0] == self.P:
            solutions = np.transpose(solutions)
        new = self.copy()
        new.objvals = np.append(new.objvals, objvals)
        new.solutions = np.append(new.solutions, solutions, axis = 0)
        return new

    def append(self, obj):
        return self.add(objvals = obj.objvals, solutions = obj.solutions)

    def computeObjvals(self, getObjval):
        new = self.copy()
        compute_ind = np.flatnonzero(np.isnan(new.objvals))
        new.objvals[compute_ind] = map(getObjval, new.solutions[compute_ind])
        return new

    def removeSuboptimal(self, objval_cutoff):
        return self.filter(self.objvals <= objval_cutoff)

    def removeInfeasible(self, isFeasible):
        return self.filter(map(isFeasible, self.solutions))

    def removeNonintegral(self):
        return self.filter(map(self.isIntegral, self.solutions))

    def removeIntegral(self):
        return self.filter(~map(self.isIntegral, self.solutions))

    def table(self):
        x = pt.PrettyTable(align = 'r', float_format = '1.4', hrules=pt.ALL)
        x.add_column("objval", self.objvals.tolist())
        x.add_column("solution", map(self.solutionString, self.solutions))
        return str(x)

    def __repr__(self):
        return self.table()

class SolutionQueue(object):

    """
    SolutionQueue is written to work faster than SolutionPool and is only used by the callback functions in risk_slim
    helper class used to create/manipulate a queue of solutions and objective values
    """

    def __init__(self, P):
        self.P = P
        self.objvals = np.empty(shape=0)
        self.solutions = np.empty(shape=(0, self.P))

    def __len__(self):
        return len(self.objvals)

    def add(self, new_objvals, new_solutions):
        n_elements = self.objvals.size
        self.objvals = np.append(self.objvals, new_objvals)
        n_elements = n_elements - self.objvals.size
        self.solutions = np.append(self.solutions, np.reshape(new_solutions, (n_elements, self.P)), axis=0)

    def get_best_objval_and_solution(self):
        idx = np.argmin(self.objvals)
        return self.objvals[idx], self.solutions[idx]

    def filter_sort_unique(self, max_objval=float('Inf')):
        # filter
        if max_objval < float('inf'):
            good_idx = self.objvals <= max_objval
            self.objvals = self.objvals[good_idx]
            self.solutions = self.solutions[good_idx]

        if len(self.objvals) > 0:
            sort_idx = np.argsort(self.objvals)
            self.objvals = self.objvals[sort_idx]
            self.solutions = self.solutions[sort_idx]

            # unique
            b = np.ascontiguousarray(self.solutions).view(
                np.dtype((np.void, self.solutions.dtype.itemsize * self.P)))
            _, unique_idx = np.unique(b, return_index=True)
            self.objvals = self.objvals[unique_idx]
            self.solutions = self.solutions[unique_idx]

    def clear(self):
        self.objvals = np.empty(shape=0)
        self.solutions = np.empty(shape=(0, self.P))


