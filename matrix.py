import random
import numpy as np


def zeros(size):
    size.reverse()

    zeros_ls = []
    product = 1
    for i in range(len(size)):
        product *= size[i]

    for z in range(product):
        zeros_ls.append(0)

    main = make_shape(size, zeros_ls)

    return main


def random_uniform(size):
    size.reverse()

    zeros_ls = []
    product = 1
    for i in range(len(size)):
        product *= size[i]

    for z in range(product):
        zeros_ls.append(random.uniform(-1, 1))

    main = make_shape(size, zeros_ls)

    return main


def get_shape(input_array):
    shape = []

    i = 0
    while True:
        try:
            sum = 0
            for j in range(len(input_array)):
                sum += 1
            shape.append(sum)

            input_array = input_array[0]

            i += 1
        except TypeError:
            break

    return shape


def flatten(lst):
    def flatten_inner(lst):
        if isinstance(lst, list):
            for l in lst:
                for y in flatten(l):
                    yield y
        else:
            yield lst

    flattened = flatten_inner(lst)
    return [val for val in flattened]


def change(arr, func):
    shape = get_shape(arr)
    flattened = flatten(arr)

    shape.reverse()

    zeros_ls = []
    product = 1
    for i in range(len(shape)):
        product *= shape[i]

    for z in range(product):
        x = func(flattened[z])
        zeros_ls.append(x)

    main = make_shape(shape, zeros_ls)
    return main


def make_shape(shape, ls):
    for idx in range(len(shape) - 1):
        main = []
        for j, elem in enumerate(ls):
            if j % shape[idx] == 0:
                main.append(ls[j:j + shape[idx]])
        ls = main

    return main


def ReLU(arr):
    def oper(x):
        if x <= 0:
            x = 0
        return x

    return change(arr, oper)

def sigmoid(arr):
    def oper(x):
        x = 1/(1+np.exp(-(x)))
        return x

    return change(arr, oper)


def derivative_sigmoid(arr):
    def oper(x):
        x = (1/(1+np.exp(-(x))))*(1-(1/(1+np.exp(-(x)))))
        return x

    return change(arr, oper)


def sum(ls):
    flattened = flatten(arr)

    total = 0
    for val in flattened:
        total += val

    return total


def scalar_multiplication(arr, num):
    shape = get_shape(arr)
    flattened = flatten(arr)

    for i, _ in enumerate(flattened):
        flattened[i] *= num

    shape.reverse()
    new = make_shape(shape, flattened)
    return new


def scalar_addition(arr, num):
    shape = get_shape(arr)
    flattened = flatten(arr)

    for i, _ in enumerate(flattened):
        flattened[i] += num

    shape.reverse()
    new = make_shape(shape, flattened)
    return new


def elementwise_addition(arr1, arr2):
    shape = get_shape(arr1)

    f1 = flatten(arr1)
    f2 = flatten(arr2)

    new = []
    for i, _ in enumerate(f1):
        new.append(f1[i] + f2[i])

    shape.reverse()
    new = make_shape(shape, new)
    return new


def elementwise_subtraction(arr1, arr2):
    shape = get_shape(arr1)

    f1 = flatten(arr1)
    f2 = flatten(arr2)

    new = []
    for i, _ in enumerate(f1):
        new.append(f1[i] - f2[i])

    shape.reverse()
    new = make_shape(shape, new)
    return new



def elementwise_multiplication(arr1, arr2):
    shape = get_shape(arr1)

    f1 = flatten(arr1)
    f2 = flatten(arr2)

    new = []
    for i, _ in enumerate(f1):
        new.append(f1[i] * f2[i])

    shape.reverse()
    new = make_shape(shape, new)
    return new



def matrix_multiplication(m1, m2):
    new = []
    for i in range(len(m1)):
        row = []
        for j in range(len(m2[0])):
            total = 0
            for k in range(len(m2)):
                total += m1[i][k] * m2[k][j]
            row.append(total)
        new.append(row)

    return new


def transpose(arr):
    new = [[0 for _ in range(len(arr))] for _ in range(len(arr[0]))]

    for i in range(len(arr)):
        for j in range(len(arr[0])):
            new[j][i] = arr[i][j]

    return new


def unflatten(ls):
    m = zeros([len(ls), 1])

    for i in range(len(ls)):
        m[i][0] = ls[i]

    return m









