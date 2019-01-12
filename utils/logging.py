from typing import List

import numpy as np
import scipy as scipy
import scipy.stats


class ExperimentLog():
    def __init__(self, series: List[float], signfigance_level: float):
        self.means = []
        self.variances = []
        self.confidences = []
        self.n = 1
        self.current_observation_num = 0
        self.series = series
        self.signfigance_level = signfigance_level

    def observe(self, value: float):
        mean = None
        variance = None
        if self.current_observation_num > len(self.means) - 1:
            self.means.append(0.0)
            self.variances.append(0.0)
            mean = 0.0
            variance = 0.0
        else:
            mean = self.means[self.current_observation_num]
            variance = self.variances[self.current_observation_num]

        delta = value - mean
        mean += delta / self.n
        variance += delta * (value - mean)

        self.means[self.current_observation_num] = mean
        self.variances[self.current_observation_num] = variance
        self.current_observation_num += 1

    def finalize_confidences(self):
        assert self.n > 1
        self.variances = [variance / (self.n - 1) for variance in
                          self.variances]

        for (mean, variance) in zip(self.means, self.variances):
            crit = scipy.stats.t.ppf(1.0 - self.signfigance_level, self.n - 1)
            width = crit * np.math.sqrt(variance) / np.math.sqrt(self.n)
            self.confidences.append(width)

    def observe_trial_end(self):
        self.n += 1
        self.current_observation_num = 0


def save(name, log: ExperimentLog, out_dir: str, unique_num: int = 0):

    out_prefix = out_dir

    if not os.path.exists(out_prefix):
        os.makedirs(out_prefix)
    filename = str(experiment_num) + "_" + str(num_trials) + "_" + name + str(unique_num) + ".csv"
    full_out_path = os.path.join(out_prefix, filename)
    if log.n > 1:
        log.finalize_confidences()
        data = np.c_[(log.series, log.means, log.variances, log.confidences)]
    else:
        data = np.c_[(log.series, log.means)]
    np.savetxt(full_out_path, data,
               fmt=["%d", "%f", "%f", "%f"],
               delimiter=",")
