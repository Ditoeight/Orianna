import itertools
import math
import threading
from numpy.random import randint, seed

EVENTS = []

seed(42)
for ii in range(0, 30):
    inst = []
    inst.append(str(ii))
    inst.append(float(randint(0, 100) / 100))
    EVENTS.append(inst)


def probability_independent_events(events, target_pct, threadcount=1):
    base_target = int(math.ceil(len(events) * target_pct))
    end_target = len(events)
    event_dict = {}
    for event in events:
        event_dict[event[0]] = [event[1]]

    threadlist = []
    full_chunk = end_target - base_target
    portion = full_chunk // threadcount
    sub_portion = full_chunk % threadcount

    if full_chunk == 0:
        return independent_event_p(event_dict, base_target, end_target)
    else:
        if sub_portion == 0:
            for threadnum in range(0, threadcount):
                bottom = (threadnum * portion) + base_target
                top = bottom + portion
                threadlist.append(threading.Thread(target=independent_event_p,
                                  args=(event_dict, bottom, top)))
        else:
            for threadnum in range(0, threadcount - 1):
                bottom = (threadnum * portion) + base_target
                top = bottom + portion
                threadlist.append(threading.Thread(target=independent_event_p,
                                  args=(event_dict, bottom, top)))
            bottom = (threadcount - 1) * portion
            top = bottom + sub_portion
            threadlist.append(threading.Thread(target=independent_event_p,
                              args=(event_dict, bottom, top)))

    for thread in threadlist:
        thread.start()

    for thread in threadlist:
        thread.join()


def independent_event_p(events_as_dict, low_target, high_target):
    this_run_sum = 0.0

    for event_lengths in range(low_target, high_target + 1):
        this_length_sum = 0.0
        for option in itertools.combinations(events_as_dict.keys(),
                                             event_lengths):
            pos = 1
            neg = 1
            for key in events_as_dict:
                if key in option:
                    pos *= events_as_dict[key][0]
                else:
                    neg *= 1 - events_as_dict[key][0]
            this_length_sum += pos * neg

        this_run_sum += this_length_sum

    return this_run_sum


if __name__ == '__main__':
    print(probability_independent_events(EVENTS, .75, 3))
