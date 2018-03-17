import numpy as np
from prettytable import PrettyTable
from debug import ipsh

# Todo
# - Coefficient Set should be composed of CoefficientElements
# Unit Tests for Coefficient Element
# Unite Tests for Coefficient Set

class CoefficientElement(object):

    _VALID_TYPES = ['I', 'C']

    def _is_integer(self, x):
        return np.array_equal(x, np.require(x, dtype = np.int_))


    def __init__(self, name, ub = 10, lb = -10, c0 = float('nan'), vtype = 'I'):

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
        assert value >= self._ub
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
            self.c0 = float('nan')
        else:
            assert np.isfinite(value)
            assert value >= 0.0
            self.c0 = float(value)




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
        return str(self.tabulate())


    def view(self):
        print(self.tabulate())


    def tabulate(self):
        s = ['%s: %s' % ('name', self._name),
             '%s: %1.1f' % ('ub', self._ub),
             '%s: %1.1f' % ('lb', self._lb),
             '%s: %1.2g' % ('c0', self._c0),
             '%s: %1.0f' % ('sign', self.sign),
             '%s: %s' % ('vtype', self._vtype)]
        return '\n'.join(s)




class CoefficientSet(object):



    def __init__(self, variable_names, **kwargs):

        self._print_flag = True
        self._check_flag = True
        self_correct_flag = True

        for name in variable_names:
            assert isinstance(name, str), 'variable_names must be a list of strings'
        assert len(variable_names) == len(set(variable_names)), 'variable_names contain elements with unique names'
        self._variable_names = list(variable_names)

        self.print_flag = kwargs.get('print_flag', self._print_flag)
        self.check_flag = kwargs.get('check_flag', self._check_flag)
        self.correct_flag = kwargs.get('correct_flag', self_correct_flag)

        ub = kwargs.get('ub', 5)
        lb = kwargs.get('lb', -5)
        C_0j = kwargs.get('C0_j', float('nan'))
        sign = kwargs.get('sign', float('nan'))
        vtype = kwargs.get('type', 'I')

        ub = self._repeat_element(value = ub, element_type = float)
        lb = self._repeat_element(value = lb, element_type = float)
        C_0j = self._repeat_element(value = C_0j, element_type = float)
        vtype = self._repeat_element(value = vtype, element_type = str)

        self._coef_elements = {}
        for name in variable_names:
            idx = variable_names.index(name)
            self._coef_elements[name] = CoefficientElement(name = name, ub = ub[idx], lb = lb[idx], c0 = C_0j[idx], vtype = vtype[idx])

        self.check_numeric_input('ub', ub)
        self.check_numeric_input('lb', lb)
        self.check_numeric_input('C_0j', C_0j)
        self.check_numeric_input('sign', sign)
        self.check_string_input('vtype', vtype)
        self._check_rep()


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
        assert len(names) == len(self), 'variable_names must contain at least %d elements' % len(self)
        assert len(names) == len(set(names)), 'variable_names contain elements with unique names'
        self._variable_names = list(names)


    def tabulate(self):
        x = PrettyTable()
        x.align = "r"
        x.add_column("variable_name", self._variable_names)
        x.add_column("vtype", self.get_field_as_list('vtype'))
        x.add_column("sign", self.get_field_as_list('sign'))
        x.add_column("lb", self.get_field_as_list('lb'))
        x.add_column("ub", self.get_field_as_list('ub'))
        x.add_column("C_0j", self.get_field_as_list('C_0j'))
        return str(x)


    def view(self):
        print(self.tabulate())


    def __len__(self):
        return self.P


    def __str__(self):
        return self.tabulate()


    def __repr__(self):
        if self.print_flag:
            return self.tabulate()

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
        else:
            assert isinstance(key, str)
            assert key in self._variable_names
            assert value.name == key

        assert isinstance(value, CoefficientElement)
        self._coef_elements[key] = value

    def set_field(self, field_name, variable_names, field_values):

        # check field name
        assert hasattr(self, field_name), 'coefficient set does not contain field named %s' % field_name


        # convert field type to proper type
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        elif isinstance(variable_names, np.ndarray):
            variable_names = variable_names.to_list()
        else:
            assert isinstance(variable_names, list)

        # convert field values to proper type
        if isinstance(field_values, np.ndarray):
            field_values = field_values.to_list()
        elif not isinstance(field_values, list):
            field_values = [field_values]
        assert isinstance(field_values, list)

        # make sure that we have the correct sizes of arrays
        n_variables = len(variable_names)
        assert n_variables >= 1, '# of variables must be >= 1'

        n_values = len(field_values)
        if n_values == 1:
            field_values = field_values * n_variables
        elif n_values > 1:
            assert n_values == n_variables, '# of values (%d) must match # of variables (%d)' % (n_values, n_variables)
        else:
            raise ValueError('# of values must be 1 or %d' % n_variables)

        # set the values
        curr_values = getattr(self, field_name)
        for vname in variable_names:
            if vname in self.variable_names:
                user_ind = variable_names.index(vname)
                self_ind = self._variable_names.index(vname)
                curr_values[self_ind] = field_values[user_ind]
            else:
                if self.print_flag:
                    print("warning: coefficient object does not contain a variable named %s" % vname)

        assert self._check_rep()


    def _repeat_element(self, value, element_type):

        if isinstance(value, np.ndarray):
            if value.size == self.P:
                value_array = value
            elif value.size == 1:
                value_array = np.repeat(value, self.P)
            else:
                raise ValueError("CoefficientSet can only be initialized using properties with 1 element or len(CoefficientSet) elements")

        elif isinstance(value, list):
            if len(value) == self.P:
                value_array = value
            elif len(value) == 1:
                value_array = [value] * self.P
            else:
                raise ValueError("CoefficientSet can only be initialized using properties with 1 element or len(CoefficientSet) elements")

        elif isinstance(value, str):
            value_array = [str(value)] * self.P

        elif isinstance(value, int):
            value_array = [int(value)] * self.P

        elif isinstance(value, float):
            value_array = [float(value)] * self.P

        else:
            raise ValueError("unknown variable type %s")

        return(value_array)



    def _check_rep(self):
        if self._check_flag:
            for name in self.variable_names:
                self.check_element(name)
        return True


    def check_string_input(self, input_name, input_value):

        if isinstance(input_value, np.ndarray):

            if input_value.size == self.P:
                setattr(self, input_name, input_value)
            elif input_value.size == 1:
                setattr(self, input_name, np.repeat(input_value, self.P))
            else:
                raise ValueError("length of %s is %d; should be %d" % (input_name, input_value.size, self.P))

        elif isinstance(input_value, list):
            if len(input_value) == self.P:
                setattr(self, input_name, np.array([str(x) for x in input_value]))
            elif len(input_value) == 1:
                setattr(self, input_name, np.repeat(input_value, self.P))
            else:
                raise ValueError("length of %s is %d; should be %d" % (input_name, len(input_value), self.P))

        elif isinstance(input_value, str):
            setattr(self, input_name, float(input_value) * np.ones(self.P))

        else:
            raise ValueError("user provided %s with an unsupported type" % input_name)


    def check_numeric_input(self, input_name, input_value):

        if isinstance(input_value, np.ndarray):

            if input_value.size == self.P:
                setattr(self, input_name, np.array(input_value))
            elif input_value.size == 1:
                setattr(self, input_name, input_value * np.ones(self.P))
            else:
                raise ValueError("length of %s is %d; should be %d" % (input_name, input_value.size, self.P))

        elif isinstance(input_value, float):
            setattr(self, input_name, float(input_value) * np.ones(self.P))

        elif isinstance(input_value, int):
            setattr(self, input_name, float(input_value) * np.ones(self.P))


        elif isinstance(input_value, list):
            if len(input_value) == self.P:
                setattr(self, input_name, np.array([float(x) for x in input_value]))
            elif len(input_value) == 1:
                setattr(self, input_name, np.array([float(x) for x in input_value]) * np.ones(self.P))
            else:
                raise ValueError("length of %s is %d; should be %d" % (input_name, len(input_value), self.P))

        else:
            raise ValueError("user provided %s with an unsupported type" % (input_name))


    def correct_element(self, name):

        idx = self.variable_names.index(name)

        if self.ub[idx] < self.lb[idx]:
            correct_lb = float(self.ub[idx])
            correct_ub = float(self.lb[idx])
            self.ub[idx] = correct_ub
            self.lb[idx] = correct_lb
            if self.print_flag:
                print "fixed issue: ub < lb for variable %s" % self.variable_names[i]

        if self.lb[idx] < 0 and self.sign[idx] > 0:
            self.lb[idx] = 0.0

        if self.ub[idx] > 0 and self.sign[idx] < 0:
            self.ub[idx] = 0.0

        if name in {'Intercept', '(Intercept)', 'intercept', '(intercept)'}:
            if self.C_0j[idx] > 0 or np.isnan(self.C_0j[idx]):
                if self.print_flag:
                    print "found intercept variable with penalty value of C_0j = %1.4f" % self.C_0j[idx]
                if self.fix_flag:
                    if self.print_flag:
                        print "setting C_0j for intercept to 0.0 to ensure that intercept is not penalized"
                    self.C_0j[idx] = 0.0


    def check_element(self, name):

        idx = self._variable_names.index(name)

        if self.ub[idx] < self.lb[idx]:
            correct_lb = float(self.ub[idx])
            correct_ub = float(self.lb[idx])
            self.ub[idx] = correct_ub
            self.lb[idx] = correct_lb
            if self.print_flag:
                print("fixed issue: ub < lb for variable %s" % name)

        if self.lb[idx] < 0 and self.sign[idx] == 1:
            self.lb[idx] = 0.0

        if self.ub[idx] > 0 and self.sign[idx] == -1:
            self.ub[idx] = 0.0

        if name in {'Intercept', '(Intercept)', 'intercept', '(intercept)'}:
            if self.C_0j[idx] > 0 or np.isnan(self.C_0j[idx]):
                if self.print_flag:
                    print("found intercept variable with penalty value of C_0j = %1.2g" % self.C_0j[idx])
                if self.correct_flag:
                    if self.print_flag:
                        print("setting C_0j for intercept to 0.0 to ensure that intercept is not penalized")
                    self.C_0j[idx] = 0.0


    def get_field_as_nparray(self, field_name):
        return np.array(getattr(self, field_name))


    def get_field_as_list(self, field_name):
        return np.array(getattr(self, field_name)).tolist()






