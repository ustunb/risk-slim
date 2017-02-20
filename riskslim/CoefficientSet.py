import numpy as np
from prettytable import PrettyTable

class CoefficientSet(object):

    def check_string_input(self, input_name, input_value):

        if type(input_value) is np.array:

            if input_value.size == self.P:
                setattr(self, input_name, input_value)
            elif input_value.size == 1:
                setattr(self, input_name, np.repeat(input_value, self.P))
            else:
                raise ValueError("length of %s is %d; should be %d" % (input_name, input_value.size, self.P))

        elif type(input_value) is str:
            setattr(self, input_name, float(input_value)*np.ones(self.P))

        elif type(input_value) is list:
            if len(input_value) == self.P:
                setattr(self, input_name, np.array([str(x) for x in input_value]))
            elif len(input_value) == 1:
                setattr(self, input_name, np.repeat(input_value, self.P))
            else:
                raise ValueError("length of %s is %d; should be %d" % (input_name, len(input_value), self.P))

        else:
            raise ValueError("user provided %s with an unsupported type" % input_name)

    def check_numeric_input(self, input_name, input_value):

        if type(input_value) is np.ndarray:

            if input_value.size == self.P:
                setattr(self, input_name, input_value)
            elif input_value.size == 1:
                setattr(self, input_name, input_value*np.ones(self.P))
            else:
                raise ValueError("length of %s is %d; should be %d" % (input_name, input_value.size, self.P))

        elif type(input_value) is float or type(input_value) is int:
            setattr(self, input_name, float(input_value)*np.ones(self.P))

        elif type(input_value) is list:
            if len(input_value) == self.P:
                setattr(self, input_name, np.array([float(x) for x in input_value]))
            elif len(input_value) == 1:
                setattr(self, input_name, np.array([float(x) for x in input_value]) * np.ones(self.P))
            else:
                raise ValueError("length of %s is %d; should be %d" % (input_name, len(input_value), self.P))

        else:
            raise ValueError("user provided %s with an unsupported type" % (input_name))

    def __init__(self, **kwargs):

        if 'variable_names' in kwargs:
            variable_names = kwargs.get('variable_names')
            P = len(variable_names)
        elif 'P' in kwargs:
            P = kwargs.get('P')
            variable_names = ["x_" + str(i) for i in range(1, P+1)]
        else:
            raise ValueError("user needs to provide 'P' or 'variable_names'")

        self.P = P
        self.variable_names = variable_names
        self.fix_flag   = kwargs.get('fix_flag', True)
        self.check_flag = kwargs.get('check_flag', True)
        self.print_flag = kwargs.get('print_flag', False)

        ub = kwargs.get('ub', 10.0 * np.ones(P))
        lb = kwargs.get('lb', -10.0 * np.ones(P))
        vtype = kwargs.get('type', ['I']*P)
        C_0j = kwargs.get('C0_j', np.nan*np.ones(P))
        sign = kwargs.get('sign', np.nan*np.ones(P))

        self.check_numeric_input('ub', ub)
        self.check_numeric_input('lb', lb)
        self.check_numeric_input('C_0j', C_0j)
        self.check_numeric_input('sign', sign)
        self.check_string_input('vtype', vtype)

        if self.check_flag: self.check_set()
        if self.print_flag: self.view()

    def __len__(self):
        return self.P

    def check_set(self):

        for i in range(0, len(self.variable_names)):

            if self.ub[i] < self.lb[i]:
                if self.print_flag:
                    print "fixed issue: ub < lb for variable %s" % self.variable_names[i]
                ub = ub[i]
                lb = lb[i]
                self.ub[i] = lb
                self.lb[i] = ub

            if self.sign[i] > 0 and self.lb[i] < 0:
                self.lb[i] = 0.0

            if self.sign[i] < 0 and self.ub[i] > 0:
                self.ub[i] = 0.0

            if self.variable_names[i] in {'Intercept','(Intercept)', 'intercept', '(intercept)'}:
                if self.C_0j[i] > 0 or np.isnan(self.C_0j[i]):
                    if self.print_flag:
                        print "found intercept variable with penalty value of C_0j = %1.4f" % self.C_0j[i]
                    if self.fix_flag:
                        if self.print_flag:
                            print "setting C_0j for intercept to 0.0 to ensure that intercept is not penalized"
                        self.C_0j[i] = 0.0

    def get_field_as_nparray(self, field_name):
        return np.array(getattr(self, field_name))

    def get_field_as_list(self, field_name):
        return np.array(getattr(self, field_name)).tolist()

    def set_field(self, field_name, variable_names, field_values):

        curr_values = getattr(self, field_name)

        if type(variable_names) is str:

            variable_names = [variable_names]
            if type(field_values) is list:
                if len(field_values) is 1:
                    pass
                else:
                    raise ValueError("user provided multiple values for single field")

            elif type(field_values) is np.ndarray:
                if len(field_values) is 1:
                    pass
                else:
                    raise ValueError("user provided multiple values for single field")
                field_values = field_values.to_list()

            else:
                field_values = [field_values]

        elif type(variable_names) is list:

            if type(field_values) is list:
                if len(field_values) != len(variable_names):
                    raise ValueError("length of variable names and values do not match")

            elif type(field_values) is np.ndarray:
                if len(field_values) == len(variable_names):
                    raise ValueError("length of variable names and values do not match")
                field_values = field_values.to_list()

            else:
                field_values = [field_values]*len(variable_names)

        for variable_name in variable_names:

            if variable_name in self.variable_names:
                user_ind = variable_names.index(variable_name)
                self_ind = self.variable_names.index(variable_name)
                curr_values[self_ind] = field_values[user_ind]
            else:
                if self.print_flag:
                    print "warning: Lset object does not contain variable with name: %s" % variable_name

        if self.check_flag: self.check_set()
        if self.print_flag: self.view()

    def view(self):
        x = PrettyTable()
        x.align = "r"
        x.add_column("variable_name", self.variable_names)
        x.add_column("vtype", self.get_field_as_list('vtype'))
        x.add_column("sign", self.get_field_as_list('sign'))
        x.add_column("lb", self.get_field_as_list('lb'))
        x.add_column("ub", self.get_field_as_list('ub'))
        x.add_column("C_0j", self.get_field_as_list('C_0j'))
        print x