#!/usr/bin/python
import math
import itertools
import numpy as np
import pandas as pd
from numba import jit


EVENTS = []
np.random.seed(42)
for ii in range(0, 10):
    inst = []
    inst.append(str(ii))
    inst.append(float(np.random.randint(50, 100)/100))
    EVENTS.append(inst)

# EVENTS = [['a', .6],
#           ['b', .7],
#           ['c', .8],
#           ['d', .9]]


def main(events, target_pct):
    # Establish targets and constants
    base_nots = 1
    end_nots = len(events) - int(math.ceil(len(events) * target_pct))
    mod = len(events) # may need to be events + 1 for evens

    probability_dict = {}
    for event in events:
        probability_dict[event[0]] = [event[1]]

    if end_nots == 0:
        product = 1
        for event in probability_dict:
            product *= probability_dict[event]
        return product

    index_dict = {}
    for ii in range(len(events)):
        index_dict[events[ii][0]] = ii

    # Make the array
    temp_list = []
    total_sum = 1
    for event in probability_dict:
        temp_list.extend(probability_dict[event])
        total_sum *= float(probability_dict[event][0])
    if len(temp_list) % 2 == 0:
        temp_list.extend([1])
    main_array = pd.DataFrame([temp_list])
    for ii in range(len(temp_list)-1):
        temp_list.append(temp_list.pop(0))
        tdf = pd.DataFrame([list(temp_list)])
        main_array = main_array.append([temp_list], ignore_index=True)
    # print(main_array)

    # Generate unique patterns
    options = main_array.index.tolist()
    for nots in range(base_nots, end_nots + 1): # Loop through not lengths
        length_sum = 0
        patterns = []
        for combination in itertools.combinations(options, nots): # Generate a combo
            combo_sum = 0
            if combination[0] != 0: # Doesn't begin with first index, break
                break
            if len(combination) == 1:
                if [0] in patterns:
                    break
                else:
                    patterns.append([0])
                    length_sum += array_math(main_array, combination)
                    break
            current_pattern = []
            for position in combination: # Loop through items in combo
                position_index = combination.index(position)
                if position_index == len(combination)-1:
                    current_pattern.append(((
                        combination[0] + mod) - combination[position_index] % mod))
                else:
                    current_pattern.append(
                        combination[position_index+1] - combination[position_index])

            if pattern_check(current_pattern, patterns):
                patterns.append(current_pattern)
                combo_sum += array_math(main_array, combination)
            length_sum += combo_sum
        total_sum += length_sum

    print(total_sum)

def pattern_check(iterable, compare_list):
    roller = np.array(iterable)
    for ii in range(len(iterable)):
        if roller.tolist() in compare_list:
            return False
        roller = np.roll(roller, 1)
    return True

def array_math(array, combination):
    total_array = np.ones(len(array))
    for ii in range(len(array)):
        input_array = array.iloc[ii].values
        if ii in combination:
            input_array = vectorized_subtraction(np.ones(len(array)), input_array)
        total_array = np.array(vectorized_multiplication(input_array, total_array))
    return np.sum(total_array)

@jit(nopython=True, parallel=True)
def vectorized_multiplication(x, y):
    return x * y

@jit(nopython=True, parallel=True)
def vectorized_subtraction(x, y):
    return x - y

@jit(nopython=True, parallel=True)
def vetorized_addition(x, y):
    return x + y

if __name__ == '__main__':
    main(EVENTS, .80)
