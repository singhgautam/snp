import numpy as np

class Curriculum:
    def get(self, length):
        raise NotImplementedError

class StepCurriculum(Curriculum):
    def __init__(self, init_num_views = 10, till=5, end_num_views=0, allow_empty_context = False):
        self.init_num_views = init_num_views
        self.end_num_views = end_num_views
        self.till = till
        self.allow_empty_context = allow_empty_context

    def get(self, length):
        upper_bound = [self.init_num_views if i < self.till else self.end_num_views for i in range(length)]
        lower_limit = 0 if self.allow_empty_context else 1
        return [0 if k == 0 else np.random.randint(low=lower_limit, high=max(lower_limit+1, k)) for k in upper_bound]

class ManualCurriculumRandom(Curriculum):
    def __init__(self, upper_bound, allow_empty_context = False):
        self.upper_bound = upper_bound
        self.allow_empty_context = allow_empty_context

    def get(self, length):
        upper_bound = self.upper_bound[:min(length, len(self.upper_bound))]
        lower_limit = 0 if self.allow_empty_context else 1
        ret = [0 if k == 0 else np.random.randint(low=lower_limit, high=max(lower_limit+1, k)) for k in upper_bound]

        # disallow first time-step with empty context
        if ret[0] == 0:
            ret[0] = 1

        return ret

class ManualCurriculumFixed(Curriculum):
    def __init__(self, upper_bound, allow_empty_context = False):
        self.upper_bound = upper_bound
        self.allow_empty_context = allow_empty_context

    def get(self, length):
        upper_bound = self.upper_bound[:min(length, len(self.upper_bound))]
        return upper_bound