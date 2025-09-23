import pytest
from src import calculator

def test_fun1():
    assert calculator.fun1(2, 3) == 5
    assert calculator.fun1(5,0) == 5
    assert calculator.fun1 (-1, 1) == 0
    assert calculator.fun1 (-1, -1) == -2


def test_fun2():
    assert calculator.fun2(2, 3) == -1
    assert calculator.fun2(5,0) == 5
    assert calculator.fun2 (-1, 1) == -2
    assert calculator.fun2 (-1, -1) == 0

def test_fun3():
    assert calculator.fun3(2, 3) == 6
    assert calculator.fun3(5,0) == 0
    assert calculator.fun3 (-1, 1) == -1
    
    assert calculator.fun3 (-1, -1) == 1

def test_fun4():
    assert calculator.fun4(2, 3, 5) == 10
    assert calculator.fun4(5,0, -1) == 4
    assert calculator.fun4 (-1, -1, -1) == -3
    
    assert calculator.fun4 (-1, -1, 100) == 98
    
def test_fun5():
    assert calculator.fun5(10, 2) == 5
    assert calculator.fun5(7, 2) == 3.5
    assert calculator.fun5(-9, 3) == -3

    assert calculator.fun5(5.5, 2) == 2.75

    with pytest.raises(ValueError, match="Division by zero is not allowed."):
        calculator.fun5(5, 0)

    with pytest.raises(ValueError, match="Both inputs must be numbers."):
        calculator.fun5("a", 2)

    with pytest.raises(ValueError, match="Both inputs must be numbers."):
        calculator.fun5(10, "b")