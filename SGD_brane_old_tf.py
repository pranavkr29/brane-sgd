#!/usr/bin/python3

import time
import pickle
import tensorflow as tf
from numpy import zeros
from multiprocessing import Pool
num_stacks = 4
real_dtype = tf.float64
complex_dtype = tf.complex128
mean = 0.0
minval = 0.0
maxval = 2.0
search_range = int(abs(mean) + abs(maxval))


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
    return tf.Variable(tf.complex(rand_real, rand_imag))


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
        return tf.cast(0.696667, dtype=complex_dtype)
    elif (i, j) == (1, 3):
        return tf.cast(0.178734, dtype=complex_dtype)
    elif (i, j) == (1, 4):
        return tf.cast(0.292304, dtype=complex_dtype)
    elif (i, j) == (2, 3):
        return tf.cast(-0.54981, dtype=complex_dtype)
    elif (i, j) == (2, 4):
        return tf.cast(0.962468, dtype=complex_dtype)
    elif (i, j) == (3, 4):
        return tf.cast(-0.506718, dtype=complex_dtype)


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
                arr[i, j] = 1
    arr = tf.convert_to_tensor(arr, dtype=complex_dtype)
    return tf.concat([arr, tf.reshape(input, [1, input.shape[0]])], 0)


def full_vars(arg_list, index):
    arg_list[index] = make_mat(arg_list[index])
    return arg_list


def E12(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z41, Z42):
    return triple_mul(Z12, Z23, Z31) + triple_mul(Z12, Z24, Z41) - triple_mul(Z21, Z13, Z32) - triple_mul(Z21, Z14, Z42)


def E13(Z12, Z13, Z14, Z21, Z23, Z31, Z32, Z34, Z41, Z43):
    return triple_mul(Z13, Z32, Z21) - triple_mul(Z13, Z34, Z41) - triple_mul(Z31, Z12, Z23) - triple_mul(Z31, Z14, Z43)


def E23(Z12, Z13, Z21, Z23, Z24, Z31, Z32, Z34, Z42, Z43):
    return triple_mul(Z23, Z31, Z12) + triple_mul(Z23, Z34, Z42) - triple_mul(Z32, Z21, Z13) - triple_mul(Z32, Z24, Z43)


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
    return tf.matmul(Z43, Z34) + C(3, 4) * complex_id(N4) + commutator(phi1, phi2)


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


def init_vars(N4):
    Z12 = Z_gauged(1, 2, N4)
    Z23 = Z_gauged(2, 3, N4)
    Z41 = Z_gauged(4, 1, N4)

    Z13 = Z(1, 3, N4)
    Z14 = Z(1, 4, N4)
    Z21 = Z(2, 1, N4)
    Z24 = Z(2, 4, N4)
    Z31 = Z(3, 1, N4)
    Z32 = Z(3, 2, N4)
    Z34 = Z(3, 4, N4)
    Z42 = Z(4, 2, N4)
    Z43 = Z(4, 3, N4)
    phi1 = phi(N4)
    phi2 = phi_gauged(N4)
    phi3 = phi(N4)

    arg_list = [Z12, Z13, Z14, Z21, Z23, Z24, Z31,
                Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3]
    var_list = [Z13, Z14, Z21, Z24, Z31, Z32, Z34, Z42, Z43, phi1, phi2, phi3]

    return arg_list, var_list


gauge_index = 13


@tf.function
def brane_potential(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi1, phi2, phi3):
    E12_sqnorm = tf.norm(
        E12(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z41, Z42)) ** 2
    E13_sqnorm = tf.norm(
        E13(Z12, Z13, Z14, Z21, Z23, Z31, Z32, Z34, Z41, Z43)) ** 2
    E23_sqnorm = tf.norm(
        E23(Z12, Z13, Z21, Z23, Z24, Z31, Z32, Z34, Z42, Z43)) ** 2

    phi2 = make_mat(phi2)

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

    return tf.math.real(E12_sqnorm + E13_sqnorm + E23_sqnorm + F12_sqnorm + F23_sqnorm + F31_sqnorm + F41_sqnorm + F42_sqnorm + F43_sqnorm + G14_sqnorm + G24_sqnorm + G34_sqnorm + G41_sqnorm + G42_sqnorm + G43_sqnorm)


def optimize_2loops(N4):
    arg_list, var_list = init_vars(N4)

    def loss():
        return brane_potential(*arg_list)

    num_cycles = 500 + search_range * 250
    num_epochs = 100
    learning_rate = 1e-4 / search_range
    momentum = 0.99

    loss_vals = [loss()]

    print('beginning_loss:', loss_vals[-1].numpy(), 'Z12:', arg_list[0].numpy(),
          'Z13:', arg_list[1].numpy(), 'Z14:', arg_list[2].numpy(), 'phi_gauge:', arg_list[gauge_index].numpy())

    for cycle in range(num_cycles):

        #         print('cycle:', cycle, "V:", loss_vals[-1])
        #         print('learning rate:', learning_rate)

        opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum)

        for _ in range(num_epochs):
            opt.minimize(loss, var_list=var_list)

        loss_vals.append(loss())

        if loss_vals[-1] < 1e-12 or loss_vals[-1] > 1e+8 or tf.math.is_nan(loss_vals[-1]):
            break
        elif loss_vals[-1] > loss_vals[-2]:
            learning_rate /= 1.1
        else:
            learning_rate = min(learning_rate * 1.1, 1e-2)

    print('end_loss:', loss_vals[-1].numpy(), 'Z12:', arg_list[0].numpy(),
          'Z13:', arg_list[1].numpy(), 'Z14:', arg_list[2].numpy(), 'phi_gauge:', arg_list[gauge_index].numpy())

    return loss_vals[-1], arg_list


def main(N4):

    start_time = time.time()  # at the beginning of the program

    variable_list = []
    N4_list = [N4 for _ in range(num_runs[N4])]
    print("N4:", N4, "number of runs:", len(N4_list))

    threshold = 1e-6
    with Pool() as pool:
        for loss, arg_list in pool.map(optimize_2loops, N4_list):
            if loss < threshold:
                variable_list.append(full_vars(arg_list, gauge_index))

    print("N4:", N4, "number of runs:", len(N4_list),
          'results_length:', len(variable_list))
    with open('variables_N4={}_{}.data'.format(N4, gauge_index - 11), 'wb') as filehandle:
        pickle.dump(variable_list, filehandle)

    end_time = time.time()  # at the beginning of the program
    print('for N4 = {}..... time taken = {} seconds'.format(
        N4, round(end_time - start_time, 2)))


num_runs = {1: 25, 2: 4, 3: 4}
N4 = 3


if __name__ == "__main__":
    main(N4)

exit()