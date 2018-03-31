import numpy as np
from prettytable import PrettyTable

# TODO
# Unit Tests for Coefficient Element
# Unit Tests for Coefficient Set

class CoefficientElement(object):

    _DEFAULT_UB = 5
    _DEFAULT_LB = -5
    _DEFAULT_c0 = float('nan')
    _DEFAULT_TYPE = 'I'
    _VALID_TYPES = ['I', 'C']

    def _is_integer(self, x):
        return np.array_equal(x, np.require(x, dtype = np.int_))


    def __init__(self, name, ub = _DEFAULT_UB, lb = _DEFAULT_LB, c0 = _DEFAULT_c0, vtype = _DEFAULT_TYPE):

        self._name = str(name)
        self._ub = float(ub)
        self._lb = float(lb)
        self._c0 = float(c0)
        self._vtype = vtype
        assert self._check_rep()


    @property
    def name(self):
        return str(self._name)


    @property
    def vtype(self):
        return str(self._vtype)


    @vtype.setter
    def vtype(self, value):
        assert isinstance(value, str)
        assert value in self._VALID_TYPES
        self._vtype = str(value)


    @property
    def ub(self):
        return float(self._ub)


    @ub.setter
    def ub(self, value):
        if hasattr(value, '__len__'):
            assert len(value) == 1
            value = value[0]
        assert value >= self._lb
        self._ub = float(value)


    @property
    def lb(self):
        return float(self._lb)


    @lb.setter
    def lb(self, value):
        if hasattr(value, '__len__'):
            assert len(value) == 1
            value = value[0]
        assert value <= self._ub
        self._lb = float(value)


    @property
    def c0(self):
        return float(self._c0)


    @c0.setter
    def c0(self, value):
        if np.isnan(value):
            self._c0 = float('nan')
        else:
            assert np.isfinite(value), 'L0 penalty for %s must either be NaN or a finite positive number' % self._name
            assert value >= 0.0, 'L0 penalty for %s must either be NaN or a finite positive number' % self._name
            self._c0 = float(value)

    @property
    def penalized(self):
        return np.isnan(self._c0) or (self._c0 > 0.0)


    @property
    def sign(self):
        if self._ub > 0.0 and self._lb >= 0.0:
            return 1
        elif self._ub <= 0.0 and self._lb < 0.0:
            return -1
        elif self._ub == 0.0 and self._lb == 0.0:
            return 0
        else:
            return float('nan')


    @sign.setter
    def sign(self, value):
        if value > 0:
            self._lb = 0.0
        elif value < 0:
            self._ub = 0.0


    def _check_rep(self):

        #name
        assert isinstance(self._name, str)
        assert len(self._name) >= 1

        #bounds
        assert np.isfinite(self.ub)
        assert np.isfinite(self.lb)
        assert self.ub >= self.lb

        # value
        assert self._vtype in self._VALID_TYPES
        assert np.isnan(self.c0) or (self.c0 >= 0.0 and np.isfinite(self.c0))

        return True


    def __repr__(self):
        return self.tabulate()


    def __str__(self):
        return self.tabulate()


    def tabulate(self):
        s = ['-' * 60,
             'variable: %s' % self._name,
             '-' * 60,
             '%s: %1.1f' % ('ub', self._ub),
             '%s: %1.1f' % ('lb', self._lb),
             '%s: %1.2g' % ('c0', self._c0),
             '%s: %1.0f' % ('sign', self.sign),
             '%s: %s' % ('vtype', self._vtype)]
        t = '\n' + '\n'.join(s) + '\n'
        return t


class CoefficientSet(object):

    _initialized = False
    _print_flag = True
    _check_flag = True
    _correct_flag = True
    _variable_names = None
    _coef_elements = dict()

    def __init__(self, variable_names, **kwargs):

        # set variables using setter methods
        self.variable_names = list(variable_names)
        self.print_flag = kwargs.get('print_flag', self._print_flag)
        self.check_flag = kwargs.get('check_flag', self._check_flag)
        self.correct_flag = kwargs.get('correct_flag', self._correct_flag)

        ub = kwargs.get('ub', CoefficientElement._DEFAULT_UB)
        lb = kwargs.get('lb', CoefficientElement._DEFAULT_LB)
        c0 = kwargs.get('c0', CoefficientElement._DEFAULT_c0)
        vtype = kwargs.get('type', CoefficientElement._DEFAULT_TYPE)

        ub = self._expand_values(value = ub)
        lb = self._expand_values(value = lb)
        c0 = self._expand_values(value = c0)
        vtype = self._expand_values(value = vtype)

        for name in variable_names:
            idx = variable_names.index(name)
            self._coef_elements[name] = CoefficientElement(name = name, ub = ub[idx], lb = lb[idx], c0 = c0[idx], vtype = vtype[idx])

        self._check_rep()
        self._initialized = True


    @property
    def P(self):
        return len(self._variable_names)


    @property
    def print_flag(self):
        return bool(self._print_flag)


    @print_flag.setter
    def print_flag(self, flag):
        self._print_flag = bool(flag)


    @property
    def correct_flag(self):
        return bool(self._correct_flag)


    @correct_flag.setter
    def correct_flag(self, flag):
        self._correct_flag = bool(flag)


    @property
    def check_flag(self):
        return bool(self._check_flag)


    @check_flag.setter
    def check_flag(self, flag):
        self._check_flag = bool(flag)


    @property
    def variable_names(self):
        return list(self._variable_names)


    @variable_names.setter
    def variable_names(self, names):
        assert isinstance(names, list), 'variable_names must be a list'
        for name in names:
            assert isinstance(name, str), 'variable_names must be a list of strings'
        assert len(names) == len(set(names)), 'variable_names contain elements with unique names'
        if self._variable_names is not None:
            assert len(names) == len(self), 'variable_names must contain at least %d elements' % len(self)
        self._variable_names = list(names)


    def index(self, name):
        assert isinstance(name, str)
        if name in self._variable_names:
            return self._variable_names.index(name)
        else:
            raise ValueError('no variable named %s in coefficient set' % name)


    def penalized_indices(self):
        return np.array(map(lambda v: self._coef_elements[v].penalized, self._variable_names))


    def tabulate(self):
        t = PrettyTable()
        t.align = "r"
        t.add_column("variable_name", self._variable_names)
        t.add_column("vtype", self.vtype)
        t.add_column("sign", self.sign)
        t.add_column("lb", self.lb)
        t.add_column("ub", self.ub)
        t.add_column("c0", self.c0)
        return str(t)


    def view(self):
        print(self.tabulate())


    def __len__(self):
        return self.P


    def __str__(self):
        return self.tabulate()


    def __repr__(self):
        if self.print_flag:
            return self.tabulate()


    def __getattr__(self, name):

        if name == 'C_0j':
            name = 'c0'

        vals = map(lambda v: getattr(self._coef_elements[v], name), self._variable_names)
        if name in ['ub', 'lb', 'c0', 'sign', 'vtype']:
            return np.array(vals)
        else:
            return list(vals)


    def __setattr__(self, name, value):
        if self._initialized:
            assert all(map(lambda e: hasattr(e, name), self._coef_elements.values()))
            attr_values = self._expand_values(value)
            for e, v in zip(self._coef_elements.values(), attr_values):
                setattr(e, name, v)
            self._check_rep()
        else:
            object.__setattr__(self, name, value)


    def __getitem__(self, key):

        if isinstance(key, int):
            assert 0 <= int(key) <= self.P
            return self._coef_elements[self._variable_names[key]]
        elif isinstance(key, str):
            return self._coef_elements[key]
        else:
            raise KeyError('invalid key')


    def __setitem__(self, key, value):

        if isinstance(key, int):
            assert 0 <= int(key) <= self.P
            key = self._variable_names[key]
        elif isinstance(key, str):
            assert isinstance(key, str)
            assert key in self._variable_names
            assert value.name == key
        else:
            raise KeyError('invalid key')

        assert isinstance(value, CoefficientElement)
        self._coef_elements[key] = value


    def _check_rep(self):

        if self._check_flag:

            assert len(self._variable_names) == len(set(self._variable_names))

            for name in self._variable_names:
                assert isinstance(name, str)
                assert len(name) >= 1
                assert self._coef_elements[name]._check_rep()

        if self._correct_flag:

            for name in self._variable_names:
                e = self._coef_elements[name]
                if name in {'Intercept', '(Intercept)', 'intercept', '(intercept)'}:
                    if e.c0 > 0 or np.isnan(e.c0):
                        if self._print_flag:
                            print("setting c0 = 0.0 to ensure that intercept is not penalized")
                        e._c0 = 0.0

        return True


    def _expand_values(self, value):

        if isinstance(value, np.ndarray):
            if value.size == self.P:
                value_array = value
            elif value.size == 1:
                value_array = np.repeat(value, self.P)
            else:
                raise ValueError("length mismatch; need either 1 or %d values" % self.P)

        elif isinstance(value, list):
            if len(value) == self.P:
                value_array = value
            elif len(value) == 1:
                value_array = [value] * self.P
            else:
                raise ValueError("length mismatch; need either 1 or %d values" % self.P)

        elif isinstance(value, str):
            value_array = [str(value)] * self.P

        elif isinstance(value, int):
            value_array = [int(value)] * self.P

        elif isinstance(value, float):
            value_array = [float(value)] * self.P

        else:
            raise ValueError("unknown variable type %s")

        return(value_array)
