import numpy as np
import collections
from warnings import warn


def extract_history(history, score, model):
    if type(score) is not list:
        score = [score]
    # Save training history
    hist = history.history
    # Count the number of epoch
    numepo = history.epoch[-1] + 1  # len(history)
    test_keys = [f"test_{i}" for i in model.metrics_names]
    score_dict = dict(zip(test_keys, score))
    score_dict.update({"epoch": range(history.epoch[0] + 1, numepo + 1)})
    hist.update(score_dict)
    hist = collections.OrderedDict(hist)
    hist.move_to_end("epoch", last=False)
    return hist, numepo


def merge_history(history1, history2):
    ep1 = history1["epoch"]
    if isinstance(ep1, range):
        epochs1 = ep1.stop - ep1.start
    elif isinstance(ep1, collections.abc.Iterable):
        epochs1 = len(ep1)
    ep2 = history2["epoch"]
    if isinstance(ep2, range):
        epochs2 = ep2.stop - ep2.start
    elif isinstance(ep2, collections.abc.Iterable):
        epochs2 = len(ep2)

    for key, value in history1.items():
        if isinstance(value, range):
            if isinstance(history2[key], range):
                history1.update({key: range(value.start, history2[key].stop)})
            else:
                warn(
                    f"{key} was skipped merging due to inconsistent dtype in history 1 & 2."
                )
        elif isinstance(value, (float, np.generic)):
            if isinstance(history2[key], (float, np.generic)):
                history1.update({key: [value] * epochs1 + [history2[key]] * epochs2})
            else:
                warn(
                    f"{key} was skipped merging due to inconsistent dtype in history 1 & 2."
                )
        else:
            val = history2[key]
            if not isinstance(val, collections.abc.Iterable):
                val = [val] * epochs2
            value.extend(val)
    return history1
