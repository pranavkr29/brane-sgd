#!/usr/bin/python3

import time
import pickle
import tensorflow as tf
from numpy import zeros
from multiprocessing import Pool
from contextlib import contextmanager
from copy import deepcopy
num_stacks = 4
real_dtype = tf.float64
complex_dtype = tf.complex128
gauge_type = 0
scale = 1.0
minval = -4.1 * scale
maxval = 4.1 * scale


def Z(i, j, N4):
    N = [1, 1, 1, N4]
    shape = (N[i - 1], N[j - 1])
    """
    Initialize real matrices randomly, and combine to make them complex
    """
    rand_real = tf.random.uniform(
        shape, minval, maxval, dtype=real_dtype)
    rand_imag = tf.random.uniform(
        shape, minval, maxval, dtype=real_dtype)
    return tf.Variable(tf.complex(rand_real, rand_imag))


def phi(N4):
    shape = (N4, N4)
    """
    Initialize real matrices randomly, and combine to make them complex
    """
    rand_real = tf.random.uniform(
        shape, minval, maxval, dtype=real_dtype)
    rand_imag = tf.random.uniform(
        shape, minval, maxval, dtype=real_dtype)
    return tf.Variable(tf.complex(rand_real, rand_imag))


def phi_gauged(N4):
    shape = [N4]
    """
    Initialize real matrices randomly, and combine to make them complex
    """
    rand_real = tf.random.uniform(
        shape, minval, maxval, dtype=real_dtype)
    rand_imag = tf.random.uniform(
        shape, minval, maxval, dtype=real_dtype)
    return tf.Variable(tf.complex(rand_real, rand_imag), name="phi_gauged")


def Z_gauged(i, j, N4):
    N = [1, 1, 1, N4]
    shape = (N[i - 1], N[j - 1])
    """
    Initialize a zeros matrix, and replace the last element with 1
    """
    rand_real = tf.Variable(tf.zeros(shape, dtype=real_dtype))
    rand_real[-1, -1].assign(1)

    return tf.constant(tf.cast(rand_real, complex_dtype))


def C(i, j):
    if (i, j) == (1, 2):
        return tf.cast(2 / 3 * scale**2, dtype=complex_dtype)
    elif (i, j) == (1, 3):
        return tf.cast(3 / 5 * scale**2, dtype=complex_dtype)
    elif (i, j) == (1, 4):
        return tf.cast(5 / 7 * scale**2, dtype=complex_dtype)
    elif (i, j) == (2, 3):
        return tf.cast(7 / 11 * scale**2, dtype=complex_dtype)
    elif (i, j) == (2, 4):
        return tf.cast(11 / 13 * scale**2, dtype=complex_dtype)
    elif (i, j) == (3, 4):
        return tf.cast(13 / 17 * scale**2, dtype=complex_dtype)


# def C(i, j):
#     if (i, j) == (1, 2):
#         return tf.cast(0.696667 * scale**2, dtype=complex_dtype)
#     elif (i, j) == (1, 3):
#         return tf.cast(0.178734 * scale**2, dtype=complex_dtype)
#     elif (i, j) == (1, 4):
#         return tf.cast(0.292304 * scale**2, dtype=complex_dtype)
#     elif (i, j) == (2, 3):
#         return tf.cast(-0.54981 * scale**2, dtype=complex_dtype)
#     elif (i, j) == (2, 4):
#         return tf.cast(0.962468 * scale**2, dtype=complex_dtype)
#     elif (i, j) == (3, 4):
#         return tf.cast(-0.506718 * scale**2, dtype=complex_dtype)


def complex_id(N4):
    return tf.cast(tf.eye(N4, N4), complex_dtype)


def commutator(a, b):
    return tf.matmul(a, b) - tf.matmul(b, a)


def triple_mul(a, b, c):
    return tf.matmul(a, tf.matmul(b, c))


@tf.function
def make_mat(input):
    N4 = input.shape[0]
    shape = (N4 - 1, N4)
    arr = zeros(shape)
    for i in range(N4 - 1):
        for j in range(N4):
            if i + j >= N4 - 1:
                if gauge_type:
                    arr[i, j] = (i + j) * scale
                else:
                    arr[i, j] = 1 * scale
    arr = tf.convert_to_tensor(arr, dtype=complex_dtype)
    return tf.concat([arr, tf.reshape(input, [1, input.shape[0]])], 0)


def saddle1(arg_list):
    [Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42,
        Z43, phi1, phi2, phi3, phi12, phi23, phi31] = arg_list
    Z21, Z31, Z32, Z41, phi12, phi23, phi31 = [val.assign(
        val * 0) if isinstance(val, tf.Variable) else val for val in [Z21, Z31, Z32, Z41, phi12, phi23, phi31]]
    return [Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31]


def saddle2(arg_list):
    [Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42,
        Z43, phi1, phi2, phi3, phi12, phi23, phi31] = arg_list
    [Z12, Z31, Z32, Z34, phi1, phi12, phi23, phi31] = [val.assign(
        val * 0) if isinstance(val, tf.Variable) else val for val in [Z12, Z31, Z32, Z34, phi1, phi12, phi23, phi31]]
    return [Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31]


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


def init_vars(N4, t1221, t2332, tzi4, tphi):

    Z12 = Z_gauged(1, 2, N4)
    Z23 = Z_gauged(2, 3, N4)
    Z34 = Z_gauged(3, 4, N4)

    Z13 = Z(1, 3, N4)
    Z14 = Z(1, 4, N4)
    Z21 = Z(2, 1, N4)
    Z24 = Z(2, 4, N4)
    Z31 = Z(3, 1, N4)
    Z32 = Z(3, 2, N4)
    Z41 = Z(4, 1, N4)
    Z42 = Z(4, 2, N4)
    Z43 = Z(4, 3, N4)
    phi1 = phi_gauged(N4)
    phi2 = phi(N4)
    phi3 = phi(N4)
    phi12 = phi(1)
    phi23 = phi(1)
    phi31 = phi(1)

    if t1221:
        Z12, Z21 = Z21, Z12
    if t2332:
        Z23, Z32 = Z32, Z23
    if tzi4 == 1:
        Z34, Z14 = Z14, Z34
    elif tzi4 == 2:
        Z34, Z24 = Z24, Z34
    elif tzi4 == 3:
        Z34, Z41 = tf.Variable(tf.transpose(Z41)), tf.transpose(Z34)
    elif tzi4 == 4:
        Z34, Z42 = tf.Variable(tf.transpose(Z42)), tf.transpose(Z34)
    elif tzi4 == 5:
        Z34, Z43 = tf.Variable(tf.transpose(Z43)), tf.transpose(Z34)
    if tphi == 1:
        phi1, phi2 = phi2, phi1
    elif tphi == 2:
        phi1, phi3 = phi3, phi1

    return [Z12, Z13, Z14, Z21, Z23, Z24, Z31,
            Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31]


@tf.function
def brane_potential(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3, phi12, phi23, phi31):

    G21_sqnorm = tf.norm(G21(Z21, Z23, Z24, Z31, Z41, phi12)) ** 2
    G12_sqnorm = tf.norm(G12(Z12, Z13, Z14, Z32, Z42, phi12)) ** 2
    G31_sqnorm = tf.norm(G31(Z21, Z31, Z32, Z34, Z41, phi31)) ** 2
    G13_sqnorm = tf.norm(G13(Z12, Z13, Z14, Z23, Z43, phi31)) ** 2
    G32_sqnorm = tf.norm(G32(Z12, Z31, Z32, Z34, Z42, phi23)) ** 2
    G23_sqnorm = tf.norm(G23(Z13, Z21, Z23, Z24, Z43, phi23)) ** 2

    phi1, phi2, phi3 = [make_mat(
        val) if val.name == 'phi_gauged:0' else val for val in [phi1, phi2, phi3]]

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
                        + G14_sqnorm + G24_sqnorm + G34_sqnorm + G41_sqnorm + G42_sqnorm + G43_sqnorm) / scale**4


def saddle_optimize(arguments):
    var_list = [val for val in arguments if isinstance(val, tf.Variable)]

    def loss():
        return brane_potential(*arguments)

    num_cycles = 500
    learning_rate = 1e-3
    momentum = 0.99

    loss_vals = [loss()]

    print('saddle beginning_loss:', loss_vals[-1], 'Z13:',
          arguments[1], 'phi12:', arguments[15])

    for cycle in range(num_cycles):

        #         print('cycle_1:', cycle, "V:", loss_vals[-1])
        #         print('learning rate:', learning_rate)

        opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum)

        for _ in range(100):
            opt.minimize(loss, var_list=var_list)

        loss_vals.append(loss())

        if loss_vals[-1] < 1e-9 or tf.math.is_nan(loss_vals[-1]):
            break
        elif loss_vals[-1] < loss_vals[-2]:
            learning_rate = min(learning_rate * 1.1, 2e-2)
        elif loss_vals[-1] > loss_vals[-2]:
            learning_rate /= 1.1

    print('saddle end_loss:', loss_vals[-1], 'Z13:',
          arguments[1], 'phi12:', arguments[15])

    return loss_vals[-1], arguments


def optimize_2loops(N4, t1221, t2332, tzi4, tphi):
    print(N4, t1221, t2332, tzi4, tphi)

    arg_list = init_vars(N4, t1221, t2332, tzi4, tphi)
    # var_list = [val for val in arg_list if type(val).__name__ == 'ResourceVariable']
    var_list = [val for val in arg_list if isinstance(val, tf.Variable)]

    def loss():
        return brane_potential(*arg_list)

    loss_vals = [loss()]

    num_cycles = 500
    learning_rate = 1 / loss_vals[0]
    momentum = 0.99

    print('beginning_loss:', loss_vals[-1], 'Z12:',
          arg_list[0], '\n phi12:', arg_list[15])

    for cycle in range(num_cycles):

        #         print('cycle_1:', cycle, "V:", loss_vals[-1])
        #         print('learning rate:', learning_rate)

        opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum)

        for _ in range(100):
            opt.minimize(loss, var_list=var_list)

        loss_vals.append(loss())

        if loss_vals[-1] < 1e-15 or tf.math.is_nan(loss_vals[-1]):
            break
        elif loss_vals[-1] < loss_vals[-2]:
            learning_rate = min(learning_rate * 1.1, 1e-2 * scale)
        elif loss_vals[-1] > loss_vals[-2]:
            learning_rate /= 1.1

    print('end_loss:', loss_vals[-1], 'Z12:',
          arg_list[0], '\n phi12:', arg_list[15])

    return loss_vals[-1], arg_list


def main(N4):

    start_time = time.time()  # at the beginning of the program

    variable_list = []
    saddle2_variables_list = []
    saddle1_variables_list = []

    params = []
    N4_list = [N4 for _ in range(num_runs[N4])]
    for num in N4_list:
        for t1221 in range(2):
            for t2332 in range(2):
                for tzi4 in range(6):
                    for tphi in range(1):
                        params.append((num, t1221, t2332, tzi4, tphi))

    print("N4:", N4, "number of runs:", len(params))

    @contextmanager
    def poolcontext(*args, **kwargs):
        pool = Pool(*args, **kwargs)
        yield pool
        pool.terminate()

    threshold = 1e-6
    with poolcontext() as pool:
        for loss, arg_list in pool.starmap(optimize_2loops, params):
            if loss < threshold:
                variable_list.append(arg_list)

        # new_variable_list = deepcopy(variable_list)

        # saddle2_init_vars = [saddle2(args) for args in new_variable_list]
        # for saddle2_loss, saddle2_arg_list in pool.map(saddle_optimize, saddle2_init_vars):
        #     if saddle2_loss < threshold:
        #         saddle2_variables_list.append(saddle2_arg_list)

        # if z_fixed != 1:
        #     saddle1_init_vars = [
        #         saddle1(args) for args in new_variable_list]
        #     for saddle1_loss, saddle1_arg_list in pool.map(saddle_optimize, saddle1_init_vars):
        #         if saddle1_loss < threshold:
        #             saddle1_variables_list.append(saddle1_arg_list)

    for args in variable_list + saddle2_variables_list + saddle1_variables_list:
        print(args[17])
    print("N4:", N4, "number of runs:", len(params),
          'results_length:', len(variable_list + saddle2_variables_list + saddle1_variables_list))
    fname = 'variables_N4={}.({})_{}.data'.format(N4, scale, gauge_type)
    with open(fname, 'wb') as filehandle:
        pickle.dump(variable_list + saddle2_variables_list +
                    saddle1_variables_list, filehandle)
    end_time = time.time()  # at the beginning of the program
    print('for N4 = {}..... time taken = {} seconds'.format(
        N4, round(end_time - start_time, 2)))
    print(">>> Dumped to {}".format(fname))


num_runs = {1: 1, 2: 2, 3: 4}
N4 = 1


if __name__ == "__main__":
    main(N4)

exit()
