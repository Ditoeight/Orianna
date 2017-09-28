import itertools
import math

EVENTS = [['a', 1.0],
          ['b', .20],
          ['c', .30],
          ['d', .40],
          ['e', .50],
          ['f', .60],
          ['g', 1.0],
          ['h', .80],
          ['i', .90]]

# EVENTS = [['a', 1],
#           ['b', 0],
#           ['c', 1]]


def probability_independent_events(events, target_pct):
    base_target = int(math.ceil(len(events) * target_pct))
    end_target = len(events)

    event_dict = {}
    for event in events:
        event_dict[event[0]] = [event[1]]

    this_run_sum = 0.0

    for event_lengths in range(base_target, end_target + 1):
        positives_options = list(
            itertools.combinations(event_dict.keys(), event_lengths))
        this_length_sum = 0.0
        for option in positives_options:
            pos = 1
            neg = 1
            for key in event_dict:
                if key in option:
                    if event_dict[key][0] == 0:
                        neg *= 1
                    pos *= event_dict[key][0]
                else:
                    neg *= 1 - event_dict[key][0]
            this_length_sum += pos * neg

        this_run_sum += this_length_sum

    return this_run_sum

if __name__ == '__main__':
    print(probability_independent_events(EVENTS, .66))