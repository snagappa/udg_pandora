#!/usr/bin/env python

class State(objet):

    def __init__(self, name, transition, condition, type_dmp, machine_files):
        self.name = name
        self.transition = transition
        self.condition = condition
        self.type_dmp = type_dmp
        self.mahcine_files = machine_files

    def __del__(self):
        print 'Burn this state ' + str(self.name)


class StateMachine(object):

    def __init__(self,state_machine_name):
        pass
        self.current_state

    def parse_state_machine(self, file_name):
        pass

    def update_enviroment(self,enviroment_parameters):
        """
        recive the information to know if the state has change,
        returns name of the state or end to finish the state machine
        """
        pass

    def get_current_state():
        pass

    def get_all_states():
        """
        returns a list with all the states to create the DMP machines needed.
        """
        pass

    def get_init_state():
        pass

    def __del__(self):
        print 'Destroy everything'
