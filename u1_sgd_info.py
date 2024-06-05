import time
import collections
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import pathos.multiprocessing as mp


class Variables:
    """Class for generating the variables and constants of the system"""

    def __init__(self, N4):
        self.N4 = N4

    def Z(self, i, j, minval=-3, maxval=3, name="Z"):
        N = [1, 1, 1, self.N4]
        shape = (N[i - 1], N[j - 1])
        """
        Initialize real matrices randomly, and combine to make them complex
        """
        rand_real = np.random.uniform(size=shape, low=minval, high=maxval)
        rand_imag = np.random.uniform(size=shape, low=minval, high=maxval)
        z = np.array(rand_real, dtype=complex)
        z.imag = rand_imag
        return tf.Variable(z, name=name, trainable=True)

    def X(self, k, a, minval=-3, maxval=3, name="X"):
        N = [1, 1, 1, self.N4]
        shape = (N[k - 1], N[a - 1])
        """
        Initialize real matrices randomly, and combine to make them complex
        """
        rand_real = np.random.uniform(size=shape, low=minval, high=maxval)
        return tf.Variable(rand_real, name=name, trainable=True)

    def Z_gauged(self, i, j, name="ZGauged"):
        N = [1, 1, 1, self.N4]
        shape = (N[i - 1], N[j - 1])
        """
        Initialize a zeros matrix, and replace the last element with 1
        """
        rand_real = np.zeros(shape, dtype=complex)
        rand_real[-1, -1] = 1
        return tf.constant(rand_real, name=name)

    def Z_U1gauged(self, i, j, minval=-3, maxval=3, name="ZU1"):
        N = [1, 1, 1, self.N4]
        shape = (N[i - 1], N[j - 1])
        rand_real = np.random.uniform(size=shape, low=minval, high=maxval)
        return tf.Variable(rand_real, name=name, trainable=True)

    def phi(self, N, minval=-3, maxval=3, name="phiV"):
        shape = (N, N)
        """
        Initialize real matrices randomly, and combine to make them complex
        """
        rand_real = np.random.uniform(size=shape, low=minval, high=maxval)
        rand_imag = np.random.uniform(size=shape, low=minval, high=maxval)
        z = np.array(rand_real, dtype=complex)
        z.imag = rand_imag
        return tf.Variable(z, name=name, trainable=True)


class Functions:
    """Class for the transformations and functions of the variables."""

    def __init__(self, N4):
        self.N4 = N4
        self.C = tf.constant(
            [[0, 2 / 3, 3 / 5, 5 / 7], [0, 0, 7 / 11, 11 / 13], [0, 0, 0, 13 / 17]],
            dtype=tf.complex128,
        )
        self.FI = tf.constant([1, 2, 3, -6], dtype=tf.complex128)
        self.lam = tf.constant(1, dtype=tf.complex128)
        self.SUSY_coeff = 0

    def commutator(self, a, b):
        return tf.linalg.matmul(a, b) - tf.linalg.matmul(b, a)

    def distance(self, a, b):
        """Sum of squares of distance between entries of two lists"""
        return tf.math.real(sum([tf.norm(a[i] - b[i]) ** 2 for i in range(len(a))]))

    def neg_vars(self, input_dict):
        """
        Negate each value in the input dictionary and return the modified dictionary.

        Args:
            input_dict (dict): The input dictionary.

        Returns:
            dict: A new dictionary with negated values.
        """
        negated_dict = {key: -value for key, value in input_dict.items()}
        return negated_dict

    def conj_vars(self, input_dict):
        """
        Complex Conjugate each value in the input dictionary and return the modified dictionary.

        Args:
            input_dict (dict): The input dictionary.

        Returns:
            dict: A new dictionary with negated values.
        """
        negated_dict = {key: tf.math.conj(value) for key, value in input_dict.items()}
        return negated_dict

    def full_vars_U1(
        self,
        Z12,
        Z13,
        Z14,
        Z21,
        Z23,
        Z24,
        Z31,
        Z32,
        Z34,
        Z41,
        Z42,
        Z43,
        X11,
        X12,
        X13,
        X21,
        X22,
        X23,
        X31,
        X32,
        X33,
        phi1,
        phi2,
        phi3,
        phi12,
        phi23,
        phi31,
    ):
        Z12, Z23, Z34, X11, X12, X13, X21, X22, X23, X31, X32, X33 = (
            tf.complex(Z12, tf.zeros_like(Z12)),
            tf.complex(Z23, tf.zeros_like(Z23)),
            tf.complex(Z34, tf.zeros_like(Z34)),
            tf.complex(X11, tf.zeros_like(X11)),
            tf.complex(X12, tf.zeros_like(X12)),
            tf.complex(X13, tf.zeros_like(X13)),
            tf.complex(X21, tf.zeros_like(X21)),
            tf.complex(X22, tf.zeros_like(X22)),
            tf.complex(X23, tf.zeros_like(X23)),
            tf.complex(X31, tf.zeros_like(X31)),
            tf.complex(X32, tf.zeros_like(X32)),
            tf.complex(X33, tf.zeros_like(X33)),
        )
        return (
            Z12,
            Z13,
            Z14,
            Z21,
            Z23,
            Z24,
            Z31,
            Z32,
            Z34,
            Z41,
            Z42,
            Z43,
            X11,
            X12,
            X13,
            X21,
            X22,
            X23,
            X31,
            X32,
            X33,
            phi1,
            phi2,
            phi3,
            phi12,
            phi23,
            phi31,
        )

    def brane_potential(
        self,
        Z12,
        Z13,
        Z14,
        Z21,
        Z23,
        Z24,
        Z31,
        Z32,
        Z34,
        Z41,
        Z42,
        Z43,
        X11,
        X12,
        X13,
        X21,
        X22,
        X23,
        X31,
        X32,
        X33,
        phi1,
        phi2,
        phi3,
        phi12,
        phi23,
        phi31,
    ):
        F12_sqnorm = tf.norm(tf.linalg.matmul(Z12, Z21) + self.C[0, 1]) ** 2
        F21_sqnorm = tf.norm(tf.linalg.matmul(Z21, Z12) + self.C[0, 1]) ** 2

        F23_sqnorm = tf.norm(tf.linalg.matmul(Z23, Z32) + self.C[1, 2]) ** 2
        F32_sqnorm = tf.norm(tf.linalg.matmul(Z32, Z23) + self.C[1, 2]) ** 2

        F31_sqnorm = tf.norm(tf.linalg.matmul(Z31, Z13) + self.C[0, 2]) ** 2
        F13_sqnorm = tf.norm(tf.linalg.matmul(Z13, Z31) + self.C[0, 2]) ** 2

        F41_sqnorm = (
            tf.norm(
                tf.linalg.matmul(Z41, Z14)
                + self.C[0, 3] * tf.eye(self.N4, dtype=tf.complex128)
                + self.commutator(phi2, phi3)
            )
            ** 2
        )
        F14_sqnorm = tf.norm(tf.linalg.matmul(Z14, Z41) + self.C[0, 3] * self.N4) ** 2
        F42_sqnorm = (
            tf.norm(
                tf.linalg.matmul(Z42, Z24)
                + self.C[1, 3] * tf.eye(self.N4, dtype=tf.complex128)
                + self.commutator(phi3, phi1)
            )
            ** 2
        )
        F24_sqnorm = tf.norm(tf.linalg.matmul(Z24, Z42) + self.C[1, 3] * self.N4) ** 2
        F43_sqnorm = (
            tf.norm(
                tf.linalg.matmul(Z43, Z34)
                + self.C[2, 3] * tf.eye(self.N4, dtype=tf.complex128)
                - self.commutator(phi2, phi1)
            )
            ** 2
        )
        F34_sqnorm = tf.norm(tf.linalg.matmul(Z34, Z43) + self.C[2, 3] * self.N4) ** 2
        G21_sqnorm = (
            tf.norm(
                tf.linalg.matmul(Z21, phi12)
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z23, Z31))
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z24, Z41))
            )
            ** 2
        )
        G12_sqnorm = (
            tf.norm(
                tf.linalg.matmul(Z12, phi12)
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z13, Z32))
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z14, Z42))
            )
            ** 2
        )
        G31_sqnorm = (
            tf.norm(
                tf.linalg.matmul(Z31, phi31)
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z32, Z21))
                - tf.math.multiply(self.lam, tf.linalg.matmul(Z34, Z41))
            )
            ** 2
        )
        G13_sqnorm = (
            tf.norm(
                tf.linalg.matmul(Z13, phi31)
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z12, Z23))
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z14, Z43))
            )
            ** 2
        )
        G32_sqnorm = (
            tf.norm(
                tf.linalg.matmul(Z32, phi23)
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z31, Z12))
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z34, Z42))
            )
            ** 2
        )
        G23_sqnorm = (
            tf.norm(
                tf.linalg.matmul(Z23, phi23)
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z21, Z13))
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z24, Z43))
            )
            ** 2
        )
        G14_sqnorm = (
            tf.norm(
                -tf.linalg.matmul(Z14, phi1)
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z12, Z24))
                - tf.math.multiply(self.lam, tf.linalg.matmul(Z13, Z34))
            )
            ** 2
        )
        G24_sqnorm = (
            tf.norm(
                -tf.linalg.matmul(Z24, phi2)
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z21, Z14))
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z23, Z34))
            )
            ** 2
        )
        G34_sqnorm = (
            tf.norm(
                -tf.linalg.matmul(Z34, phi3)
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z31, Z14))
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z32, Z24))
            )
            ** 2
        )
        G41_sqnorm = (
            tf.norm(
                -tf.linalg.matmul(phi1, Z41)
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z42, Z21))
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z43, Z31))
            )
            ** 2
        )
        G42_sqnorm = (
            tf.norm(
                -tf.linalg.matmul(phi2, Z42)
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z43, Z32))
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z41, Z12))
            )
            ** 2
        )
        G43_sqnorm = (
            tf.norm(
                -tf.linalg.matmul(phi3, Z43)
                - tf.math.multiply(self.lam, tf.linalg.matmul(Z41, Z13))
                + tf.math.multiply(self.lam, tf.linalg.matmul(Z42, Z23))
            )
            ** 2
        )
        V_gauge = (
            (tf.norm(X11 - X21) ** 2) * tf.norm(Z12) ** 2
            + (tf.norm(X21 - X11) ** 2) * tf.norm(Z21) ** 2
            + (tf.norm(X11 - X31) ** 2) * tf.norm(Z13) ** 2
            + (tf.norm(X31 - X11) ** 2) * tf.norm(Z31) ** 2
            + (tf.norm(X21 - X31) ** 2) * tf.norm(Z23) ** 2
            + (tf.norm(X31 - X21) ** 2) * tf.norm(Z32) ** 2
            + (tf.norm(X11) ** 2) * tf.norm(Z14) ** 2
            + (tf.norm(-X11) ** 2) * tf.norm(Z41) ** 2
            + (tf.norm(X21) ** 2) * tf.norm(Z24) ** 2
            + (tf.norm(-X21) ** 2) * tf.norm(Z42) ** 2
            + (tf.norm(X31) ** 2) * tf.norm(Z34) ** 2
            + (tf.norm(-X31) ** 2) * tf.norm(Z43) ** 2
            + (tf.norm(X12 - X22) ** 2) * tf.norm(Z12) ** 2
            + (tf.norm(X22 - X12) ** 2) * tf.norm(Z21) ** 2
            + (tf.norm(X12 - X32) ** 2) * tf.norm(Z13) ** 2
            + (tf.norm(X32 - X12) ** 2) * tf.norm(Z31) ** 2
            + (tf.norm(X22 - X32) ** 2) * tf.norm(Z23) ** 2
            + (tf.norm(X32 - X22) ** 2) * tf.norm(Z32) ** 2
            + (tf.norm(X12) ** 2) * tf.norm(Z14) ** 2
            + (tf.norm(-X12) ** 2) * tf.norm(Z41) ** 2
            + (tf.norm(X22) ** 2) * tf.norm(Z24) ** 2
            + (tf.norm(-X22) ** 2) * tf.norm(Z42) ** 2
            + (tf.norm(X32) ** 2) * tf.norm(Z34) ** 2
            + (tf.norm(-X32) ** 2) * tf.norm(Z43) ** 2
            + (tf.norm(X13 - X23) ** 2) * tf.norm(Z12) ** 2
            + (tf.norm(X23 - X13) ** 2) * tf.norm(Z21) ** 2
            + (tf.norm(X13 - X33) ** 2) * tf.norm(Z13) ** 2
            + (tf.norm(X33 - X13) ** 2) * tf.norm(Z31) ** 2
            + (tf.norm(X23 - X33) ** 2) * tf.norm(Z23) ** 2
            + (tf.norm(X33 - X23) ** 2) * tf.norm(Z32) ** 2
            + (tf.norm(X13) ** 2) * tf.norm(Z14) ** 2
            + (tf.norm(-X13) ** 2) * tf.norm(Z41) ** 2
            + (tf.norm(X23) ** 2) * tf.norm(Z24) ** 2
            + (tf.norm(-X23) ** 2) * tf.norm(Z42) ** 2
            + (tf.norm(X33) ** 2) * tf.norm(Z34) ** 2
            + (tf.norm(-X33) ** 2) * tf.norm(Z43) ** 2
        )

        D1_sqnorm = (
            tf.norm(
                tf.norm(Z12) ** 2
                + tf.norm(Z13) ** 2
                + tf.norm(Z14) ** 2
                - tf.norm(Z21) ** 2
                - tf.norm(Z31) ** 2
                - tf.norm(Z41) ** 2
                - self.FI[0]
            )
            ** 2
        )
        D2_sqnorm = (
            tf.norm(
                tf.norm(Z21) ** 2
                + tf.norm(Z23) ** 2
                + tf.norm(Z24) ** 2
                - tf.norm(Z12) ** 2
                - tf.norm(Z32) ** 2
                - tf.norm(Z42) ** 2
                - self.FI[1]
            )
            ** 2
        )
        D3_sqnorm = (
            tf.norm(
                tf.norm(Z31) ** 2
                + tf.norm(Z32) ** 2
                + tf.norm(Z34) ** 2
                - tf.norm(Z13) ** 2
                - tf.norm(Z23) ** 2
                - tf.norm(Z43) ** 2
                - self.FI[2]
            )
            ** 2
        )
        D4_sqnorm = (
            tf.norm(
                tf.linalg.matmul(Z41, tf.linalg.adjoint(Z41))
                + tf.linalg.matmul(Z42, tf.linalg.adjoint(Z42))
                + tf.linalg.matmul(Z43, tf.linalg.adjoint(Z43))
                - tf.linalg.matmul(tf.linalg.adjoint(Z14), Z14)
                - tf.linalg.matmul(tf.linalg.adjoint(Z24), Z24)
                - tf.linalg.matmul(tf.linalg.adjoint(Z34), Z34)
                + self.commutator(phi1, tf.linalg.adjoint(phi1))
                + self.commutator(phi2, tf.linalg.adjoint(phi2))
                + self.commutator(phi3, tf.linalg.adjoint(phi3))
                - self.FI[3] * tf.eye(self.N4, dtype=tf.complex128) / self.N4
            )
            ** 2
        )

        return (
            2
            * (
                G21_sqnorm
                + G12_sqnorm
                + G31_sqnorm
                + G13_sqnorm
                + G32_sqnorm
                + G23_sqnorm
                + F12_sqnorm
                + F21_sqnorm
                + F23_sqnorm
                + F32_sqnorm
                + F31_sqnorm
                + F13_sqnorm
                + F14_sqnorm
                + F24_sqnorm
                + F34_sqnorm
                + F41_sqnorm
                + F42_sqnorm
                + F43_sqnorm
                + G14_sqnorm
                + G24_sqnorm
                + G34_sqnorm
                + G41_sqnorm
                + G42_sqnorm
                + G43_sqnorm
            )
            + V_gauge
            + 1 / 2 * (D1_sqnorm + D2_sqnorm + D3_sqnorm + D4_sqnorm)
            + self.SUSY_coeff * tf.norm(phi1) ** 2
        )

    def uniquify(
        self, solutions_set, dval_list, dist_threshold=1e-4, use_symmetry=False
    ):
        unique_solutions_dval_set = []
        for j, arg_dict in enumerate(solutions_set):
            flipper = 1
            # check if solution exists in unique_list or not
            for i, (sol_dict, _) in enumerate(unique_solutions_dval_set):
                # print("sol_dict:", sol_dict)
                # print("arg_list:", arg_list)

                a_dist = self.distance(
                    self.B_invariant(**arg_dict), self.B_invariant(**sol_dict)
                )
                b_dist = self.distance(
                    self.B_invariant(**arg_dict),
                    self.B_invariant(**self.neg_vars(sol_dict)),
                )
                c_dist = self.distance(
                    self.B_invariant(**arg_dict),
                    self.B_invariant(**self.conj_vars(sol_dict)),
                )
                d_dist = self.distance(
                    self.B_invariant(**arg_dict),
                    self.B_invariant(**self.neg_vars(self.conj_vars(sol_dict))),
                )

                if use_symmetry:
                    if (
                        a_dist < dist_threshold
                        or b_dist < dist_threshold
                        or c_dist < dist_threshold
                        or d_dist < dist_threshold
                    ):
                        arg_loss, sol_loss = self.brane_potential(
                            *self.full_vars_U1(**arg_dict)
                        ), self.brane_potential(*self.full_vars_U1(**sol_dict))
                        # print("distance:", min(a_dist, b_dist, c_dist, d_dist))
                        if tf.math.real(arg_loss) < tf.math.real(sol_loss):
                            unique_solutions_dval_set[i] = (arg_dict, dval_list[j])
                        flipper *= 0
                        break
                    break
                else:
                    if a_dist < dist_threshold:
                        arg_loss, sol_loss = self.brane_potential(
                            *self.full_vars_U1(**arg_dict)
                        ), self.brane_potential(*self.full_vars_U1(**sol_dict))
                        # print("distance:", min(a_dist, b_dist, c_dist, d_dist))
                        if tf.math.real(arg_loss) < tf.math.real(sol_loss):
                            unique_solutions_dval_set[i] = (arg_dict, dval_list[j])
                        flipper *= 0
                        break
                    break

            if flipper:
                unique_solutions_dval_set.append((arg_dict, dval_list[j]))
                print("no of unique solutions:", len(unique_solutions_dval_set))

        return unique_solutions_dval_set

    def B_invariant(
        self,
        Z12,
        Z13,
        Z14,
        Z21,
        Z23,
        Z24,
        Z31,
        Z32,
        Z34,
        Z41,
        Z42,
        Z43,
        X11,
        X12,
        X13,
        X21,
        X22,
        X23,
        X31,
        X32,
        X33,
        phi1,
        phi2,
        phi3,
        phi12,
        phi23,
        phi31,
    ):
        """Receives combined variables to produce a list of independent markers"""
        B = []
        """Append phi12, phi12, phi23, phi31 first"""

        B.append(phi12[0, 0])
        B.append(phi23[0, 0])
        B.append(phi31[0, 0])

        """Adding Tr(phi1), Tr(phi2), and Tr(phi3)"""
        B.append(tf.linalg.trace(phi1))
        B.append(tf.linalg.trace(phi2))
        B.append(tf.linalg.trace(phi3))

        """The next 3 are u7, u8, and u9 respectively"""
        # B.append(triple_mul(Z12, Z24, Z41)[0, 0])
        # B.append(triple_mul(Z13, Z34, Z41)[0, 0])
        # B.append(triple_mul(Z23, Z34, Z42)[0, 0])
        # for n in range(1, N4 + 1):
        #     B.append(tf.linalg.matmul(tf.linalg.matmul(Z14, (mat_power(phi1, n))),
        #                        tf.linalg.matmul(mat_power(phi2, N4 - n), Z41))[0, 0])
        #     B.append(tf.linalg.matmul(tf.linalg.matmul(Z24, (mat_power(phi2, n))),
        #                        tf.linalg.matmul(mat_power(phi3, N4 - n), Z42))[0, 0])
        #     B.append(tf.linalg.matmul(tf.linalg.matmul(Z34, (mat_power(phi3, n))),
        #                        tf.linalg.matmul(mat_power(phi1, N4 - n), Z43))[0, 0])
        return B

    def write_markers(self, full_arg_dict, data_dict):
        sol_array = self.full_vars_U1(**full_arg_dict)
        loss = self.brane_potential(*sol_array)
        data_dict["loss"].append(tf.math.real(loss).numpy())
        B = self.B_invariant(*sol_array)
        for i in range(len(B)):
            data_dict[str(i) + "-real"].append(tf.math.real(B[i]).numpy())
            data_dict[str(i) + "-imag"].append(tf.math.imag(B[i]).numpy())

    def write_hyperparameters_U1(self, data_dict):
        data_dict["C12_real"].append(tf.math.real(self.C[0, 1]).numpy())
        data_dict["C12_imag"].append(tf.math.imag(self.C[0, 1]).numpy())
        data_dict["C13_real"].append(tf.math.real(self.C[0, 2]).numpy())
        data_dict["C13_imag"].append(tf.math.imag(self.C[0, 2]).numpy())
        data_dict["C14_real"].append(tf.math.real(self.C[0, 3]).numpy())
        data_dict["C14_imag"].append(tf.math.imag(self.C[0, 3]).numpy())
        data_dict["C23_real"].append(tf.math.real(self.C[1, 2]).numpy())
        data_dict["C23_imag"].append(tf.math.imag(self.C[1, 2]).numpy())
        data_dict["C24_real"].append(tf.math.real(self.C[1, 3]).numpy())
        data_dict["C24_imag"].append(tf.math.imag(self.C[1, 3]).numpy())
        data_dict["C34_real"].append(tf.math.real(self.C[2, 3]).numpy())
        data_dict["C34_imag"].append(tf.math.imag(self.C[2, 3]).numpy())

        for i in range(len(self.FI)):
            val = self.FI[i]
            # print(val)
            data_dict["FI-" + str(i)].append(tf.math.real(val).numpy())

    def write_markers_U1(self, full_arg_dict, data_dict, dval):
        sol_array = self.full_vars_U1(**full_arg_dict)
        loss = self.brane_potential(*sol_array)
        data_dict["loss"].append(tf.math.real(loss).numpy())
        data_dict["dval"].append(tf.math.real(dval).numpy())
        B = self.B_invariant(*sol_array)
        for i in range(len(B)):
            data_dict[str(i) + "-real"].append(tf.math.real(B[i]).numpy())
            data_dict[str(i) + "-imag"].append(tf.math.imag(B[i]).numpy())

        self.write_hyperparameters_U1(data_dict)

    def write_full_solution_U1(self, full_arg_dict, data_dict, dval):
        sol_array = self.full_vars_U1(**full_arg_dict)
        loss = self.brane_potential(*sol_array)
        data_dict["loss"].append(tf.math.real(loss).numpy())
        data_dict["dval"].append(tf.math.real(dval).numpy())
        for i in range(len(sol_array)):
            val = sol_array[i]
            # print(val)
            data_dict[str(i) + "-real"].append(tf.math.real(val[0, 0]).numpy())
            data_dict[str(i) + "-imag"].append(tf.math.imag(val[0, 0]).numpy())

        self.write_hyperparameters_U1(data_dict)


class Solver:
    """Class for initializing and running the optimizer."""

    def __init__(self, N4):
        self.N4 = N4
        self.functions = Functions(self.N4)

    def run_optimization(
        self, var_dict, arg_dict, optimizer_class, n_steps, **optimizer_kwargs
    ):
        optimizer = optimizer_class(**optimizer_kwargs)
        all_args = {**var_dict, **arg_dict}

        def loss():
            return self.functions.brane_potential(
                *self.functions.full_vars_U1(**all_args)
            )

        # Initialize a list to store gradients
        gradients = [tf.Variable(tf.zeros_like(var)) for var in var_dict.values()]

        @tf.function
        def opt_cycle0(opt, loss, var_list, n_steps):
            for _ in tf.range(n_steps):
                with tf.GradientTape() as tape:
                    current_loss = loss()
                grads = tape.gradient(current_loss, var_list)
                opt.apply_gradients(zip(grads, var_list))

                # Accumulate gradients
                for i in range(len(var_list)):
                    gradients[i].assign_add(grads[i])

        opt_cycle0(optimizer, loss, list(var_dict.values()), n_steps)

        # Calculate the final loss value
        loss_value = loss()

        # Calculate the L2 norm of accumulated gradients
        gradient_norm = tf.sqrt(
            sum([tf.reduce_sum(tf.square(tf.abs(g))) for g in gradients])
        )

        return tf.math.real(loss_value), var_dict, arg_dict, gradient_norm

    def init_run(self, n_steps, learning_rate=1e-3):
        vars = Variables(self.N4)
        minmaxval = 15
        var_dict = {
            "Z12": vars.Z_U1gauged(
                1, 2, minval=-minmaxval, maxval=minmaxval, name="Z12"
            ),
            "Z23": vars.Z_U1gauged(
                2, 3, minval=-minmaxval, maxval=minmaxval, name="Z23"
            ),
            "Z34": vars.Z_U1gauged(
                3, 4, minval=-minmaxval, maxval=minmaxval, name="Z34"
            ),
            "Z13": vars.Z(1, 3, minval=-minmaxval, maxval=minmaxval, name="Z13"),
            "Z14": vars.Z(1, 4, minval=-minmaxval, maxval=minmaxval, name="Z14"),
            "Z21": vars.Z(2, 1, minval=-minmaxval, maxval=minmaxval, name="Z21"),
            "Z24": vars.Z(2, 4, minval=-minmaxval, maxval=minmaxval, name="Z24"),
            "Z31": vars.Z(3, 1, minval=-minmaxval, maxval=minmaxval, name="Z31"),
            "Z32": vars.Z(3, 2, minval=-minmaxval, maxval=minmaxval, name="Z32"),
            "Z41": vars.Z(4, 1, minval=-minmaxval, maxval=minmaxval, name="Z41"),
            "Z42": vars.Z(4, 2, minval=-minmaxval, maxval=minmaxval, name="Z42"),
            "Z43": vars.Z(4, 3, minval=-minmaxval, maxval=minmaxval, name="Z43"),
            "X11": vars.X(1, 1, minval=-minmaxval, maxval=minmaxval, name="X11"),
            "X12": vars.X(1, 2, minval=-minmaxval, maxval=minmaxval, name="X12"),
            "X13": vars.X(1, 3, minval=-minmaxval, maxval=minmaxval, name="X13"),
            "X21": vars.X(2, 1, minval=-minmaxval, maxval=minmaxval, name="X21"),
            "X22": vars.X(2, 2, minval=-minmaxval, maxval=minmaxval, name="X22"),
            "X23": vars.X(2, 3, minval=-minmaxval, maxval=minmaxval, name="X23"),
            "X31": vars.X(3, 1, minval=-minmaxval, maxval=minmaxval, name="X31"),
            "X32": vars.X(3, 2, minval=-minmaxval, maxval=minmaxval, name="X32"),
            "X33": vars.X(3, 3, minval=-minmaxval, maxval=minmaxval, name="X33"),
            "phi1": vars.phi(self.N4, minval=-minmaxval, maxval=minmaxval, name="phi1"),
            "phi2": vars.phi(self.N4, minval=-minmaxval, maxval=minmaxval, name="phi2"),
            "phi3": vars.phi(self.N4, minval=-minmaxval, maxval=minmaxval, name="phi3"),
            "phi12": vars.phi(1, minval=-minmaxval, maxval=minmaxval, name="phi12"),
            "phi23": vars.phi(1, minval=-minmaxval, maxval=minmaxval, name="phi23"),
            "phi31": vars.phi(1, minval=-minmaxval, maxval=minmaxval, name="phi31"),
        }
        arg_dict = {}

        opt = tf.keras.optimizers.SGD

        for _ in tqdm(range(6)):
            loss_val, var_dict, arg_dict, gradient_norm = self.run_optimization(
                var_dict, arg_dict, opt, n_steps, learning_rate=learning_rate
            )

            print(
                "loss:",
                loss_val.numpy(),
                "dval:",
                gradient_norm.numpy(),
                "phi31:",
                var_dict["phi31"][0][0].numpy(),
                "lr:",
                learning_rate,
            )

            if gradient_norm < 1:
                # learning_rate *= 0.8
                opt = tf.keras.optimizers.Adam

            if loss_val < 1e-9 or gradient_norm < 1e-8 or tf.math.is_nan(loss_val):
                break

        return loss_val, {**var_dict, **arg_dict}, gradient_norm


def main(N4, num_runs, n_steps, learning_rate):
    def solve_task(args):
        N4, n_steps, learning_rate = args
        return Solver(N4).init_run(n_steps, learning_rate)

    pool = mp.Pool()  # Create a processing pool

    # Create a list of argument tuples for the solver function
    args_list = [(N4, n_steps, learning_rate)] * num_runs

    # Use `map` to parallelize the task and get results
    results = pool.map(solve_task, args_list)

    # Close the pool to release resources
    pool.close()
    pool.join()

    solutions_list, dval_list = [], []
    for i in results:
        print("dval:", i[-1], "loss:", i[0])
        if i[-1] < 1e-6:  # Keeping only low derivative critical solutions
            solutions_list.append(i[1])
            dval_list.append(i[-1])

    functions = Functions(N4)
    markers = collections.defaultdict(list)
    solutions_to_print = collections.defaultdict(list)
    if len(solutions_list) > 0:
        unique_solutions_dval_set = functions.uniquify(solutions_list, dval_list)
        unique_solutions_list = [item[0] for item in unique_solutions_dval_set]

        s = functions.SUSY_coeff
        t = time.time()
        fname = f"U1info_variables_N4={N4}_s={s}_t={t}"
        fname_sol = f"U1info_solutions_N4={N4}_s={s}_t={t}"

        # with open(fname_sol + ".data", "wb") as filehandle:
        #     pickle.dump(unique_solutions_list, filehandle)

        for sol_dict, dval in unique_solutions_dval_set:
            functions.write_markers_U1(sol_dict, markers, dval)
            functions.write_full_solution_U1(sol_dict, solutions_to_print, dval)

            # functions.write_markers_U1(functions.neg_vars(sol_dict), markers, dval)

            # functions.write_markers_U1(functions.conj_vars(sol_dict), markers, dval)

            # functions.write_markers_U1(
            #     functions.conj_vars(functions.neg_vars(sol_dict)), markers, dval
            # )

        pd.DataFrame.from_dict(data=markers, orient="columns").to_csv(
            fname + ".csv", header=True, index=False
        )
        pd.DataFrame.from_dict(data=solutions_to_print, orient="columns").to_csv(
            fname_sol + ".csv", header=True, index=False
        )
    else:
        unique_solutions_list = []

    print("number of runs:", num_runs, "results_length:", len(unique_solutions_list))


if __name__ == "__main__":
    start_time = time.time()  # at the beginning of the program

    N4 = 1
    num_runs = 36
    n_steps = 6e4
    learning_rate = 2e-4
    functions = Functions(N4)
    fi = functions.FI
    c = functions.C
    result = (1 / 2) * (fi[0] ** 2 + fi[1] ** 2 + fi[2] ** 2 + fi[3] ** 2) + 4 * (
        c[0, 1] ** 2
        + c[0, 2] ** 2
        + c[0, 3] ** 2
        + c[1, 2] ** 2
        + c[1, 3] ** 2
        + c[2, 3] ** 2
    )
    print('fi:', fi)
    print("c:", c)
    print(result)
    # main(N4, num_runs, n_steps, learning_rate)
    end_time = time.time()  # at the end of the program
    print(f"for N4 = {N4}..... time taken = {round(end_time - start_time, 2)} seconds")
    exit()
