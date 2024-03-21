'''
Soft Activation Function from TEL (where gamma = smooth_step_param)
        0, if t <= -gamma/2
S(t) =  -2/(gamma**2) * t^3 + 3/(2*gamma)* t + 1/2, if -gama/2 <= t <= gamma/2
        1, if t >= gamma/2
where gamma >= 0

smooth_step_param = 1/gamma, where gamma is the parameter defined in the TEL paper (default=1.0)
Same implementation of soft activations from TEL implementation:
https://github.com/google-research/google-research/blob/master/tf_trees/neural_trees_helpers.cc
'''


def soft_activation(v, smooth_step_param):
    if v <= -0.5 / smooth_step_param:
        out = 0.
    elif v >= 0.5 / smooth_step_param:
        out = 1.
    else:
        x = smooth_step_param * v + 0.5
        x_squared = x * x
        x_cubed = x_squared * x
        out = -2. * x_cubed + 3. * x_squared
    return out


def soft_activation_derivative(v, smooth_step_param):
    if abs(v) <= (0.5 / smooth_step_param):
        x = smooth_step_param * v + 0.5
        x_squared = x * x
        out = 6. * smooth_step_param * (-x_squared + x)
    else:
        out = 0.
    return out


class SplitDecision:
    def __init__(self, feature, split_at):
        self.feature = feature
        self.split_at = split_at

    def __str__(self):
        return "Split Decision(Feature: {}, split_at: {})".format(self.feature, self.split_at)
