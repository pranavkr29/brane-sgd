#!/usr/bin/python3

import os
import time
import pickle
import tensorflow as tf
import numpy as np
from copy import deepcopy
import collections
import matplotlib.pyplot as plt
import pandas as pd
# num_stacks = 4
real_dtype = tf.float64
complex_dtype = tf.complex128
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def C(i, j):
    if (i, j) == (1, 2):
        return tf.cast(2 / 3, dtype=complex_dtype)
    elif (i, j) == (1, 3):
        return tf.cast(3 / 5, dtype=complex_dtype)
    elif (i, j) == (1, 4):
        return tf.cast(5 / 7, dtype=complex_dtype)
    elif (i, j) == (2, 3):
        return tf.cast(7 / 11, dtype=complex_dtype)
    elif (i, j) == (2, 4):
        return tf.cast(11 / 13, dtype=complex_dtype)
    elif (i, j) == (3, 4):
        return tf.cast(13 / 17, dtype=complex_dtype)


def eq26(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31):
    [val.assign(tf.math.multiply(val, scale)) if isinstance(val, tf.Variable) else tf.math.multiply(val, scale)
     for val in [Z13, Z14, Z23, Z24, Z31, Z32, Z41, Z42]]
    [val.assign(tf.math.multiply(val, tf.math.pow(scale, 2))) if isinstance(val, tf.Variable)
     else tf.math.multiply(val, tf.math.pow(scale, 2)) for val in [phi3, phi12]]
    return [Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31]


def complex_id(N4):
    return tf.cast(tf.eye(N4, N4), complex_dtype)


def commutator(a, b):
    return tf.matmul(a, b) - tf.matmul(b, a)


def triple_mul(a, b, c):
    return tf.matmul(a, tf.matmul(b, c))


def F12(Z12, Z21):
    return tf.matmul(Z12, Z21) + C(1, 2)


def F23(Z23, Z32):
    return tf.matmul(Z23, Z32) + C(2, 3)


def F31(Z13, Z31):
    return tf.matmul(Z31, Z13) + C(1, 3)


def F41(Z14, Z41, phi2, phi3):
    N4 = Z41.shape[0]
    return tf.matmul(Z41, Z14) + C(1, 4) * complex_id(N4) + commutator(phi2, phi3)


def F42(Z24, Z42, phi1, phi3):
    N4 = Z42.shape[0]
    return tf.matmul(Z42, Z24) + C(2, 4) * complex_id(N4) + commutator(phi3, phi1)


def F43(Z34, Z43, phi1, phi2):
    N4 = Z43.shape[0]
    return tf.matmul(Z43, Z34) + C(3, 4) * complex_id(N4) - commutator(phi2, phi1)


def G21(Z21, Z23, Z24, Z31, Z41, phi12):
    return tf.matmul(Z21, phi12) + tf.matmul(Z23, Z31) + tf.matmul(Z24, Z41)


def G12(Z12, Z13, Z14, Z32, Z42, phi12):
    return tf.matmul(Z12, phi12) + tf.matmul(Z13, Z32) + tf.matmul(Z14, Z42)


def G31(Z21, Z31, Z32, Z34, Z41, phi31):
    return tf.matmul(Z31, phi31) + tf.matmul(Z32, Z21) - tf.matmul(Z34, Z41)


def G13(Z12, Z13, Z14, Z23, Z43, phi31):
    return tf.matmul(Z13, phi31) + tf.matmul(Z12, Z23) + tf.matmul(Z14, Z43)


def G32(Z12, Z31, Z32, Z34, Z42, phi23):
    return tf.matmul(Z32, phi23) + tf.matmul(Z31, Z12) + tf.matmul(Z34, Z42)


def G23(Z13, Z21, Z23, Z24, Z43, phi23):
    return tf.matmul(Z23, phi23) + tf.matmul(Z21, Z13) + tf.matmul(Z24, Z43)


def G14(Z12, Z13, Z14, Z24, Z34, phi1):
    return -tf.matmul(Z14, phi1) + tf.matmul(Z12, Z24) - tf.matmul(Z13, Z34)


def G24(Z14, Z21, Z23, Z24, Z34, phi2):
    return -tf.matmul(Z24, phi2) + tf.matmul(Z21, Z14) + tf.matmul(Z23, Z34)


def G34(Z14, Z24, Z31, Z32, Z34, phi3):
    return -tf.matmul(Z34, phi3) + tf.matmul(Z31, Z14) + tf.matmul(Z32, Z24)


def G41(Z21, Z31, Z41, Z42, Z43, phi1):
    return -tf.matmul(phi1, Z41) + tf.matmul(Z42, Z21) + tf.matmul(Z43, Z31)


def G42(Z12, Z32, Z41, Z42, Z43, phi2):
    return -tf.matmul(phi2, Z42) + tf.matmul(Z43, Z32) + tf.matmul(Z41, Z12)


def G43(Z13, Z23, Z41, Z42, Z43, phi3):
    return -tf.matmul(phi3, Z43) - tf.matmul(Z41, Z13) + tf.matmul(Z42, Z23)


def reverse_get_vars(arg_list):
    """Given that we only store the original set of solutions, the tf.Variable and tf.tensor
    structure should be preserved, except in the phi_gauged one(s). So just look for that, separate
    it into 2, convert the bottom into a variable, and you're done"""

    # arg_list = [tf.Variable(val) if not tf.equal(tf.math.real(
    #     val[0, 0]), 0.) and i != 0 and i != 4 else val for (i, val) in enumerate(arg_list)]

    for i in [12, 13, 14]:
        if not isinstance(arg_list[i], tf.Variable):
            N4 = arg_list[i].shape[1]
            top_phi, arg_list[i] = tf.split(arg_list[i], [N4 - 1, 1], 0)
            arg_list[i] = tf.Variable(arg_list[i], name="phi_gauged")
            arg_list.append(top_phi)

    return arg_list


def neg_vars(arg_list):
    return [-arg for arg in arg_list]


def conj_vars(arg_list):
    return [tf.math.conj(arg) for arg in arg_list]


def saddle1(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31, top_phi):
    [val.assign(val * 0, read_value=True) for val in [Z21, Z31, Z32,
                                                      Z41, phi12, phi23, phi31] if isinstance(val, tf.Variable)]
    return [Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31, top_phi]


def saddle2(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31, top_phi):
    [val.assign(val * 0, read_value=True) for val in [Z21, Z31, Z32,
                                                      Z34, phi1, phi12, phi23, phi31] if isinstance(val, tf.Variable)]
    return [Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31, top_phi]


def full_vars(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31, top_phi):
    phi1, phi2, phi3 = [make_mat(val, top_phi) if val.name ==
                        '1:0' or val.name == '2:0' else val for val in [phi1, phi2, phi3]]
    return [Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31]


def mat_power(a, n):
    if n == 0:
        return complex_id(a.shape[0])
    elif n == 1:
        return tf.math.pow(a, n)
    else:
        b = a
        for _ in range(1, n):
            b = tf.matmul(b, a)
        return b


def B_invariant(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31):
    N4 = phi1.shape[0]
    B = []
    """Append phi12, phi12, phi23, phi31 first"""

    B.append(tf.math.divide(phi12, 1.)[0, 0])
    B.append(tf.math.divide(phi23, 1.)[0, 0])
    B.append(tf.math.divide(phi31, 1.)[0, 0])

    """Adding Tr(phi1), Tr(phi2), and Tr(phi3)"""
    B.append(tf.math.divide(tf.linalg.trace(phi1), 1.))
    B.append(tf.math.divide(tf.linalg.trace(phi2), 1.))
    B.append(tf.math.divide(tf.linalg.trace(phi3), 1.))

    """The next 3 are u7, u8, and u9 respectively"""
    B.append(triple_mul(Z12, Z24, Z41)[0, 0])
    B.append(triple_mul(Z13, Z34, Z41)[0, 0])
    B.append(triple_mul(Z23, Z34, Z42)[0, 0])
    # for n in range(1, N4 + 1):
    #     B.append(tf.matmul(tf.matmul(Z14, (mat_power(phi1, n))),
    #                        tf.matmul(mat_power(phi2, N4 - n), Z41))[0, 0])
    #     B.append(tf.matmul(tf.matmul(Z24, (mat_power(phi2, n))),
    #                        tf.matmul(mat_power(phi3, N4 - n), Z42))[0, 0])
    #     B.append(tf.matmul(tf.matmul(Z34, (mat_power(phi3, n))),
    #                        tf.matmul(mat_power(phi1, N4 - n), Z43))[0, 0])
    return B


def write_markers(arg_list, data_dict, loss):
    data_dict['loss'].append(loss.numpy())
    B = B_invariant(*arg_list)
    for i in range(len(B)):
        data_dict[str(i) + '-real'].append(tf.math.real(B[i]).numpy())
        data_dict[str(i) + '-imag'].append(tf.math.imag(B[i]).numpy())


def brane_potential(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31, top_phi):

    Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31 = full_vars(
        Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31, top_phi)

    G21_sqnorm = tf.norm(G21(Z21, Z23, Z24, Z31, Z41, phi12)) ** 2
    G12_sqnorm = tf.norm(G12(Z12, Z13, Z14, Z32, Z42, phi12)) ** 2
    G31_sqnorm = tf.norm(G31(Z21, Z31, Z32, Z34, Z41, phi31)) ** 2
    G13_sqnorm = tf.norm(G13(Z12, Z13, Z14, Z23, Z43, phi31)) ** 2
    G32_sqnorm = tf.norm(G32(Z12, Z31, Z32, Z34, Z42, phi23)) ** 2
    G23_sqnorm = tf.norm(G23(Z13, Z21, Z23, Z24, Z43, phi23)) ** 2

    F12_sqnorm = tf.norm(F12(Z12, Z21)) ** 2
    F23_sqnorm = tf.norm(F23(Z23, Z32)) ** 2
    F31_sqnorm = tf.norm(F31(Z13, Z31)) ** 2
    F41_sqnorm = tf.norm(F41(Z14, Z41, phi2, phi3)) ** 2
    F42_sqnorm = tf.norm(F42(Z24, Z42, phi1, phi3)) ** 2
    F43_sqnorm = tf.norm(F43(Z34, Z43, phi1, phi2)) ** 2

    G14_sqnorm = tf.norm(G14(Z12, Z13, Z14, Z24, Z34, phi1)) ** 2
    G24_sqnorm = tf.norm(G24(Z14, Z21, Z23, Z24, Z34, phi2)) ** 2
    G34_sqnorm = tf.norm(G34(Z14, Z24, Z31, Z32, Z34, phi3)) ** 2
    G41_sqnorm = tf.norm(G41(Z21, Z31, Z41, Z42, Z43, phi1)) ** 2
    G42_sqnorm = tf.norm(G42(Z12, Z32, Z41, Z42, Z43, phi2)) ** 2
    G43_sqnorm = tf.norm(G43(Z13, Z23, Z41, Z42, Z43, phi3)) ** 2

    return tf.math.real(G21_sqnorm + G12_sqnorm + G31_sqnorm + G13_sqnorm + G32_sqnorm + G23_sqnorm
                        + F12_sqnorm + F23_sqnorm + F31_sqnorm + F41_sqnorm + F42_sqnorm + F43_sqnorm
                        + G14_sqnorm + G24_sqnorm + G34_sqnorm + G41_sqnorm + G42_sqnorm + G43_sqnorm)


def brane_potential_joined(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31):

    G21_sqnorm = tf.norm(G21(Z21, Z23, Z24, Z31, Z41, phi12)) ** 2
    G12_sqnorm = tf.norm(G12(Z12, Z13, Z14, Z32, Z42, phi12)) ** 2
    G31_sqnorm = tf.norm(G31(Z21, Z31, Z32, Z34, Z41, phi31)) ** 2
    G13_sqnorm = tf.norm(G13(Z12, Z13, Z14, Z23, Z43, phi31)) ** 2
    G32_sqnorm = tf.norm(G32(Z12, Z31, Z32, Z34, Z42, phi23)) ** 2
    G23_sqnorm = tf.norm(G23(Z13, Z21, Z23, Z24, Z43, phi23)) ** 2

    F12_sqnorm = tf.norm(F12(Z12, Z21)) ** 2
    F23_sqnorm = tf.norm(F23(Z23, Z32)) ** 2
    F31_sqnorm = tf.norm(F31(Z13, Z31)) ** 2
    F41_sqnorm = tf.norm(F41(Z14, Z41, phi2, phi3)) ** 2
    F42_sqnorm = tf.norm(F42(Z24, Z42, phi1, phi3)) ** 2
    F43_sqnorm = tf.norm(F43(Z34, Z43, phi1, phi2)) ** 2

    G14_sqnorm = tf.norm(G14(Z12, Z13, Z14, Z24, Z34, phi1)) ** 2
    G24_sqnorm = tf.norm(G24(Z14, Z21, Z23, Z24, Z34, phi2)) ** 2
    G34_sqnorm = tf.norm(G34(Z14, Z24, Z31, Z32, Z34, phi3)) ** 2
    G41_sqnorm = tf.norm(G41(Z21, Z31, Z41, Z42, Z43, phi1)) ** 2
    G42_sqnorm = tf.norm(G42(Z12, Z32, Z41, Z42, Z43, phi2)) ** 2
    G43_sqnorm = tf.norm(G43(Z13, Z23, Z41, Z42, Z43, phi3)) ** 2

    return tf.math.real(G21_sqnorm + G12_sqnorm + G31_sqnorm + G13_sqnorm + G32_sqnorm + G23_sqnorm
                        + F12_sqnorm + F23_sqnorm + F31_sqnorm + F41_sqnorm + F42_sqnorm + F43_sqnorm
                        + G14_sqnorm + G24_sqnorm + G34_sqnorm + G41_sqnorm + G42_sqnorm + G43_sqnorm)


def saddle_optimize(arguments, solution):
    var_list = [val for val in arguments if isinstance(val, tf.Variable)]

    sol_list = [solution, neg_vars(solution), conj_vars(
        solution), neg_vars(conj_vars(solution))]

    def loss():
        V0 = brane_potential(*arguments)
        for sols in sol_list:
            ex = tf.math.real((tf.math.tanh(
                5 * (sum([tf.norm(arguments[i] - sols[i])**2 for i in range(len(sols))]) - 1)) + 1) / 2)
            V0 = ex * V0 + 5 * (1 - ex)
        return V0

    loss_vals = [loss()]

    num_cycles = 60
    learning_rate = min(1e-3, 1 / loss_vals[0])
    momentum = 0.99

    print('saddle beginning_loss:', loss_vals[-1], 'Z12:',
          arguments[0], '\n phi31:', solution[-2])

    for cycle in range(num_cycles):

        #         print('cycle_1:', cycle, "V:", loss_vals[-1])
        #         print('learning rate:', learning_rate)

        opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum)

        @tf.function
        def opt_cycle1():
            for _ in tf.range(1e+4):
                opt.minimize(loss, var_list=var_list)

        opt_cycle1()

        loss_vals.append(loss())

        if tf.math.is_nan(loss_vals[-1]) or loss_vals[-1] < 1e-17 or cycle > int(num_cycles / 2) and loss_vals[-1] > 1e-3:
            break
        elif loss_vals[-1] < loss_vals[-2]:
            learning_rate = min(learning_rate * 2, 1e-2)
        elif loss_vals[-1] > loss_vals[-2]:
            learning_rate /= 1.1

    print('saddle end_loss:', loss_vals[-1], 'Z12:',
          arguments[0], '\n phi31:', arguments[-2])

    return loss_vals[-1], arguments


def optimize_2loops(arguments):

    var_list = [val for val in arguments if isinstance(val, tf.Variable)]

    def loss():
        return brane_potential(*arguments)

    loss_vals = [loss()]

    num_cycles = 100
    learning_rate = min(1e-3, 1 / loss_vals[0])
    momentum = 0.99

    print('beginning_loss:', loss_vals[-1], 'Z12:',
          arguments[0], '\n phi31:', arguments[-2])

    for cycle in range(num_cycles):

        # print('cycle_1:', cycle, "V:", loss_vals[-1])
        # print('learning rate:', learning_rate)

        opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum)

        @tf.function
        def opt_cycle1():
            for _ in tf.range(1e+3):
                opt.minimize(loss, var_list=var_list)

        opt_cycle1()

        loss_vals.append(loss())

        if tf.math.is_nan(loss_vals[-1]) or loss_vals[-1] < 1e-21 or cycle > int(num_cycles / 2) and loss_vals[-1] > 1e-3:
            break
        elif loss_vals[-1] < loss_vals[-2]:
            learning_rate = min(learning_rate * 2, 1e-2)
        elif loss_vals[-1] > loss_vals[-2]:
            learning_rate /= 1.1

    print('end_loss:', loss_vals[-1], 'Z12:',
          arguments[0], '\n phi31:', arguments[-2])

    return loss_vals[-1], arguments


from random import sample


def main_distill(N4):

    result_dir = '../input/brane-sgd-variables/'
    result_dir = './Good Results/'
    data_filename = 'variables_N4={}.combined'.format(N4)

    start_time = time.time()  # at the beginning of the program
    with open(result_dir + data_filename + '.data', 'rb') as filehandle:
        # read the data as binary data stream
        loaded_variables = pickle.load(filehandle)
        unique_list = []

    def distance(a, b):
        return tf.math.real(sum([tf.norm(a[i] - b[i])**2 for i in range(len(a))]))

    dist_threshold = 1e-4
    # loaded_variables = sample(loaded_variables, min(100, len(loaded_variables)))
    # loaded_variables = loaded_variables[:int(len(loaded_variables)/2)]

    print("N4:", N4, 'total samples:', len(loaded_variables))

    # traverse for all elements
    for arg_list in loaded_variables:
        flipper = 1
        # check if exists in unique_list or not
        for (i, sol_list) in enumerate(unique_list):
            a_dist = distance(B_invariant(*arg_list), B_invariant(*sol_list))
            b_dist = distance(B_invariant(*arg_list),
                              B_invariant(*neg_vars(sol_list)))
            c_dist = distance(B_invariant(*arg_list),
                              B_invariant(*conj_vars(sol_list)))
            d_dist = distance(B_invariant(*arg_list),
                              B_invariant(*neg_vars(conj_vars(sol_list))))
            if a_dist < dist_threshold or b_dist < dist_threshold or c_dist < dist_threshold or d_dist < dist_threshold:
                arg_loss, sol_loss = brane_potential_joined(
                    *arg_list), brane_potential_joined(*sol_list)
                print('distance:', min(a_dist, b_dist, c_dist, d_dist))
                if arg_loss < sol_loss:
                    unique_list[i] = arg_list
                flipper *= 0
                break

        if flipper:
            unique_list.append(arg_list)
            print(len(unique_list))

    for sols in unique_list:
        print('loss:', brane_potential_joined(*sols), 'phi31:', sols[-1])

    dist_matrix = [[distance(a, b) for a in unique_list]
                   for b in unique_list]
    dist_matrix = np.asarray(dist_matrix)
    plt.imshow(dist_matrix)
    plt.savefig('distance_{}.png'.format(data_filename), dpi=len(unique_list))
    # plt.gray()
    plt.show()

    print("N4:", N4, "number of runs:", len(loaded_variables),
          'results_length:', len(unique_list))

    with open(data_filename + 'distilled.data', 'wb') as filehandle:
        pickle.dump(unique_list, filehandle)

    end_time = time.time()  # at the beginning of the program
    print('for N4 = {}..... time taken = {} seconds'.format(
        N4, round(end_time - start_time, 2)))


def main_saddle(N4):

    result_dir = '../input/brane-sgd-variables/'
    result_dir = './'
    data_filename = 'variables_N4={}'.format(N4)

    solutions_list = []
    markers = collections.defaultdict(list)

    start_time = time.time()  # at the beginning of the program
    with open(result_dir + data_filename + '.data', 'rb') as filehandle:
        # read the data as binary data stream
        loaded_variables = pickle.load(filehandle)

    # loaded_variables = sample(loaded_variables, min(2, len(loaded_variables)))
    # loaded_variables = loaded_variables[:int(len(loaded_variables)/2)]

    print("N4:", N4, 'total samples:', len(loaded_variables))

    threshold = 1e-6
    for arg_list in loaded_variables:

        arg_list = reverse_get_vars(list(arg_list))
        saddle_init_args = saddle2(*deepcopy(arg_list))

        sol_loss, sol_list = saddle_optimize(saddle_init_args, arg_list)
        if sol_loss < threshold:
            full_sols = full_vars(*sol_list)

            print('.....')

            """Only ever append the original solution to the .data file.
            It preserves the tensor and variable structure"""

            solutions_list.append(full_sols)
            write_markers(full_sols, markers, sol_loss)

            write_markers(neg_vars(full_sols), markers, sol_loss)

            write_markers(conj_vars(full_sols), markers, sol_loss)

            write_markers(conj_vars(neg_vars(full_sols)),
                          markers, sol_loss)

    print('*' * 8)
    for sol_list in solutions_list:
        print(sol_list[-1])

    print("N4:", N4, "number of runs:", len(loaded_variables),
          'results_length:', len(solutions_list))

    with open(data_filename + 'rerun.data', 'wb') as filehandle:
        pickle.dump(solutions_list, filehandle)

    (pd.DataFrame.from_dict(data=markers, orient='columns').to_csv(
        data_filename + '.csv'.format(N4), header=True, index=False))

    print('csv results_length:', len(markers['loss']))

    end_time = time.time()  # at the beginning of the program
    print('for N4 = {}..... time taken = {} seconds'.format(
        N4, round(end_time - start_time, 2)))
    print(">>> Dumped to {}.csv".format(data_filename))


def main_write(N4):

    result_dir = '../input/brane-sgd-variables/'
    result_dir = './Good Results/data/'
    data_filename = 'variables_N4={}'.format(N4)

    solutions_list = []
    markers = collections.defaultdict(list)

    start_time = time.time()  # at the beginning of the program
    with open(result_dir + data_filename + '.data', 'rb') as filehandle:
        # read the data as binary data stream
        loaded_variables = pickle.load(filehandle)

    # print(*loaded_variables[1], sep='\n')
    # loaded_variables = loaded_variables[:int(len(loaded_variables) / 2)]
    # loaded_variables = [loaded_variables[1]]

    print("N4:", N4, 'total samples:', len(loaded_variables))

    threshold = 1e-6
    threshold1 = 1e-15
    for arg_list in loaded_variables:

        # arg_list = interfere(list(arg_list))
        # print(*arg_list, sep='\n')

        sol_loss = brane_potential_joined(*arg_list)
        print(sol_loss)
        print(arg_list[12], arg_list[13], arg_list[14], arg_list[-1])

        # if sol_loss > threshold:
        #     sol_loss, sol_list = optimize_2loops(arg_list)

        if sol_loss < threshold:
            # full_sols = full_vars(*arg_list)
            full_sols = arg_list

            # print('loss:', sol_loss, 'phi31:', full_sols[-1])
            print('.....')

            """Only ever append the original solution to the .data file.
            It preserves the tensor and variable structure"""

            solutions_list.append(full_sols)
            write_markers(full_sols, markers, sol_loss)

            write_markers(neg_vars(full_sols), markers, sol_loss)

            write_markers(conj_vars(full_sols), markers, sol_loss)

            write_markers(conj_vars(neg_vars(full_sols)),
                          markers, sol_loss)

    for sol_list in solutions_list:
        print(sol_list[-1])

    print("N4:", N4, "number of runs:", len(loaded_variables),
          'results_length:', len(solutions_list))

    # with open(data_filename + 'rerun.data', 'wb') as filehandle:
    #     pickle.dump(solutions_list, filehandle)

    (pd.DataFrame.from_dict(data=markers, orient='columns').to_csv(
        data_filename + '.csv'.format(N4), header=True, index=False))

    print('csv results_length:', len(markers['loss']))

    end_time = time.time()  # at the beginning of the program
    print('for N4 = {}..... time taken = {} seconds'.format(
        N4, round(end_time - start_time, 2)))
    print(">>> Dumped to {}.csv".format(data_filename))


if __name__ == "__main__":
    N4 = 3
    main_distill(N4)
    exit()
