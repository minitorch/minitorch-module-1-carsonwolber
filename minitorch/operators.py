"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
def mul(a: float, b:float) -> float:
  return a * b 
# - id
def id(a): # type: ignore
  return a
# - add
def add(a: float, b:float) -> float:
  return a + b
# - neg
def neg(a: float) -> float:
  return -a
# - lt
def lt(a: float, b:float):
  return float(a < b)
# - eq
def eq(a: float, b:float):
  return float(a == b)
# - max
def max(a: float, b:float):
  if a > b :
    return a
  return b 
# - is_close
def is_close(a: float, b: float):
  return float(abs(a - b) < .01)
# - sigmoid
def sigmoid(x: float) -> float:
  return 1.0 / (1.0 + math.e ** -x)
# - relu
def relu(x: float) -> float:
  return float(max(x, 0.0))
# - log
def log(x: float) -> float:
  return math.log(x, math.e)
# - exp
def exp(x:float):
  return math.e ** x
# - log_back
def log_back(a: float, b: float):
  return 1/a * b
# - inv
def inv(a: float) -> float:
    if a != 0:
        return 1/a
    else:
      return 1.0

# - inv_back
def inv_back(a: float, b:float):
  (-1/a**2) * b 
# - relu_back
def relu_back(x: float, y:float):
  if max(x,0) == x:
    return y
  return 0
  
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce

def map(fn: Callable[[float], float], ls: Iterable[float]):
  return[fn(x) for x  in ls]

def zipWith(f: Callable[[float, float], float], x: Iterable[float], y: Iterable[float]):
  return [f(i,j) for i,j in zip(x,y)]

def reduce(f: Callable[[float, float], float], x: Iterable[float]):
  if x: 
    acc = iter(x)
    a = next(acc)
    for i in acc:
      a = f(i, a)
    return a
  else:
    return 0

def negList(x: Iterable[float]):
  return map(neg, x)

def addLists(x: Iterable[float], y:Iterable[float]):
  return zipWith(add, x, y)

def sum(x: Iterable[float]):
  return reduce(add, x)

def prod(x: Iterable[float]):
  return reduce(mul, x)

# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

