import itertools
import math
from numpy.random import randint, seed

EVENTS = []

seed(42)
for ii in range(0,30):
    inst = []
    inst.append(str(ii))
    inst.append(float(randint(0, 100)/100))
    EVENTS.append(inst)


def probability_independent_events(events, target_pct):
    base_target = int(math.ceil(len(events) * target_pct))
    end_target = len(events)
    event_dict = {}
    for event in events:
        event_dict[event[0]] = [event[1]]

    this_run_sum = 0.0

    for event_lengths in range(base_target, end_target + 1):
        this_length_sum = 0.0
        for option in itertools.combinations(event_dict.keys(), event_lengths):
            pos = 1
            neg = 1
            for key in event_dict:
                if key in option:
                    pos *= event_dict[key][0]
                else:
                    neg *= 1 - event_dict[key][0]
            this_length_sum += pos * neg

        this_run_sum += this_length_sum

    return this_run_sum

if __name__ == '__main__':
    print(probability_independent_events(EVENTS, .1))
    # print(EVENTS)
