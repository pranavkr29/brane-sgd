#!/usr/bin/python3

# adapted from https://github.com/jankrepl/mildlyoverfitted/blob/master/mini_tutorials/custom_optimizer_in_pytorch/src.py


import numpy as np
import torch
from torch.optim import Adam, SGD, ASGD, LBFGS, RMSprop
from tqdm import tqdm

def Z(i, j, N4, minval=-1, maxval=1):
    N = [1, 1, 1, N4]
    shape = (N[i - 1], N[j - 1])
    """
    Initialize real matrices randomly, and combine to make them complex
    """
    rand_real = np.random.uniform(size=shape, low=minval, high=maxval)
    rand_imag = np.random.uniform(size=shape, low=minval, high=maxval)
    z = np.array(rand_real, dtype=complex)
    z.imag = rand_imag
    return torch.tensor(z, requires_grad=True)


def phi(N4, minval=-1, maxval=1):
    shape = (N4, N4)
    """
    Initialize real matrices randomly, and combine to make them complex
    """
    rand_real = np.random.uniform(size=shape, low=minval, high=maxval)
    rand_imag = np.random.uniform(size=shape, low=minval, high=maxval)
    z = np.array(rand_real, dtype=complex)
    z.imag = rand_imag
    return torch.tensor(z, requires_grad=True)


def phi_gauged(N4, minval=-1, maxval=1):
    shape = (1, N4)
    """
    Initialize real matrices randomly, and combine to make them complex
    """
    rand_real = np.random.uniform(size=shape, low=minval, high=maxval)
    rand_imag = np.random.uniform(size=shape, low=minval, high=maxval)
    z = np.array(rand_real, dtype=complex)
    z.imag = rand_imag
    return torch.tensor(z, requires_grad=True)


def phi_constant(N4):
    shape = (N4 - 1, N4)
    rand_real = np.ones(shape)
    rand_imag = np.zeros(shape)
    z = np.array(rand_real, dtype=complex)
    z.imag = rand_imag
    return torch.tensor(z, requires_grad=False)



def Z_gauged(i, j, N4):
    N = [1, 1, 1, N4]
    shape = (N[i - 1], N[j - 1])
    """
    Initialize a zeros matrix, and replace the last element with 1
    """
    rand_real = np.zeros(shape, dtype = complex)
    rand_real[-1, -1] = 1
    return torch.tensor(rand_real, requires_grad=False)


def commutator(a,b):
    return torch.mm(a, b) - torch.mm(b, a)


def brane_potential(Z12, Z13, Z14, Z21, Z23, Z24, Z31, Z32, Z34, Z41, Z42, Z43, phi_gauged, phi2, phi3, phi12, phi23, phi31, phi_constant, C):
    N4 = Z14.shape[1]
    phi1 = torch.cat((phi_constant, phi_gauged), dim=0)

    F12_sqnorm = torch.norm(torch.mm(Z12, Z21) + C[0,1])**2
    F23_sqnorm = torch.norm(torch.mm(Z23, Z32) + C[1,2])**2
    F31_sqnorm = torch.norm(torch.mm(Z31, Z13) + C[0,2])**2
    F41_sqnorm = torch.norm(torch.mm(Z41, Z14) + C[0,3] * torch.eye(N4, dtype=torch.complex128) + commutator(phi2, phi3))**2
    F42_sqnorm = torch.norm(torch.mm(Z42, Z24) + C[1,3] * torch.eye(N4, dtype=torch.complex128) + commutator(phi3, phi1))**2
    F43_sqnorm = torch.norm(torch.mm(Z43, Z34) + C[2,3] * torch.eye(N4, dtype=torch.complex128) - commutator(phi2, phi1))**2
    G21_sqnorm = torch.norm(torch.mm(Z21, phi12) + torch.mm(Z23, Z31) + torch.mm(Z24, Z41))**2
    G12_sqnorm = torch.norm(torch.mm(Z12, phi12) + torch.mm(Z13, Z32) + torch.mm(Z14, Z42))**2
    G31_sqnorm = torch.norm(torch.mm(Z31, phi31) + torch.mm(Z32, Z21) - torch.mm(Z34, Z41))**2
    G13_sqnorm = torch.norm(torch.mm(Z13, phi31) + torch.mm(Z12, Z23) + torch.mm(Z14, Z43))**2
    G32_sqnorm = torch.norm(torch.mm(Z32, phi23) + torch.mm(Z31, Z12) + torch.mm(Z34, Z42))**2
    G23_sqnorm = torch.norm(torch.mm(Z23, phi23) + torch.mm(Z21, Z13) + torch.mm(Z24, Z43))**2
    G14_sqnorm = torch.norm(-torch.mm(Z14, phi1) + torch.mm(Z12, Z24) - torch.mm(Z13, Z34))**2
    G24_sqnorm = torch.norm(-torch.mm(Z24, phi2) + torch.mm(Z21, Z14) + torch.mm(Z23, Z34))**2
    G34_sqnorm = torch.norm(-torch.mm(Z34, phi3) + torch.mm(Z31, Z14) + torch.mm(Z32, Z24))**2
    G41_sqnorm = torch.norm(-torch.mm(phi1, Z41) + torch.mm(Z42, Z21) + torch.mm(Z43, Z31))**2
    G42_sqnorm = torch.norm(-torch.mm(phi2, Z42) + torch.mm(Z43, Z32) + torch.mm(Z41, Z12))**2
    G43_sqnorm = torch.norm(-torch.mm(phi3, Z43) - torch.mm(Z41, Z13) + torch.mm(Z42, Z23))**2
    
    return (G21_sqnorm + G12_sqnorm + G31_sqnorm + G13_sqnorm + G32_sqnorm + G23_sqnorm 
            + F12_sqnorm + F23_sqnorm + F31_sqnorm + F41_sqnorm + F42_sqnorm + F43_sqnorm 
            + G14_sqnorm + G24_sqnorm + G34_sqnorm + G41_sqnorm + G42_sqnorm + G43_sqnorm)


def run_optimization(var_dict, arg_dict, C, optimizer_class, n_iter, **optimizer_kwargs):

    optimizer = optimizer_class(var_dict.values(), **optimizer_kwargs)
    all_args = {**var_dict,**arg_dict}

    path = np.empty((n_iter + 1, len(var_dict)), dtype=complex)
    # path[0, :] = list(var_dict.values())

    for i in tqdm(range(1, n_iter + 1)):
        optimizer.zero_grad()
        loss = brane_potential(**all_args, C=C)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(var_dict.values(), 1e+6)
        optimizer.step()
        print(loss, var_dict['Z13'])
        # path[i, :] = var_args.detach().numpy()

    return path


if __name__ == "__main__":
    N4=2
    C = torch.tensor([[0, 2/3, 3/5, 5/7],[0, 0, 7/11, 11/13],[0, 0, 0, 13/17]], dtype=torch.complex128, requires_grad=False)
    var_dict = {'Z13': Z(1, 3, N4, minval=0, maxval=1), 'Z14': Z(1, 4, N4), 'Z21': Z(2, 1, N4),
                'Z24': Z(2, 4, N4), 'Z31': Z(3, 1, N4), 
                'Z32': Z(3, 2, N4), 'Z41': Z(4, 1, N4), 'Z42': Z(4, 2, N4), 
                'Z43': Z(4, 3, N4), 'phi_gauged': phi_gauged(N4),'phi2': phi(N4), 
                'phi3': phi(N4), 'phi12': phi(1), 'phi23':phi(1), 'phi31': phi(1)}
    arg_dict = {'Z12':Z_gauged(1, 2, N4), 'Z23': Z_gauged(2, 3, N4), 'Z34': Z_gauged(3, 4, N4), 'phi_constant': phi_constant(N4)}

    n_iter = 10000

    path_sgd = run_optimization(var_dict, arg_dict, C, SGD, n_iter, lr=1e-2, momentum=0.99)
    # path_adam = run_optimization(var_dict, arg_dict, C, Adam, n_iter, lr=1e-2)
    # path_RMS = run_optimization(var_dict, arg_dict, C, RMSprop, n_iter, lr=1e-1, momentum=0.0)


    print(arg_dict, C)
    # print(path_sgd[-1], C, sep='\n')
    
    quit()
