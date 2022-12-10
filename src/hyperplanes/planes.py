from __future__ import annotations

import copy
from typing import Union, List, Tuple

import numpy


class Hyperplane:
    """
    Oblique hyperplane of the form:
        a1f1 + a2f2 + ... + amfm <= b
    where a1, ..., am, and b are scalars. Alternatively, formulated as ax <= b.

    Attributes:
        coefficients (numpy.ndarray): Coefficients of the hyperplane (the a).
        bound (float): Bound of the hyperplane (the b).
    """

    def __init__(self, coefficients: Union[List[float], numpy.ndarray], bound: float):
        """
        An oblique (multivariate) premise
        Args:
            coefficients: Coefficients a1, ..., am
            bound: Bound scalar b
        """
        self.coefficients = coefficients if isinstance(coefficients, numpy.ndarray) else numpy.array(coefficients)
        self.bound = bound
        self.dim = self.coefficients.size

    def __hash__(self):
        return hash(str(self.coefficients.tolist() + [self.bound]))

    def __eq__(self, other):
        return isinstance(other, Hyperplane) and\
               (self.coefficients == other.coefficients).all() and \
               self.bound == other.bound

    def __repr__(self):
        return f"coefficients: {self.coefficients}\nbound: {self.bound})"

    def __str__(self):
        return f"Hyperplane\n\tcoefficients: {self.coefficients.tolist()}\n\tbound: {self.bound}"

    def __len__(self):
        """
        Length of the premise as its dimensionality
        Returns:
            The number of coefficients of this hyperplane.
        """
        return self.coefficients.size

    def __getitem__(self, item):
        if item <= self.coefficients.size:
            return self.coefficients[item]
        else:
            raise ValueError(f"Expected value in [0, {self.coefficients.size}], {item} found")

    def __setitem__(self, key, item):
        if key > len(self):
            raise ValueError(f"Expected value in [0, {len(self)}], {key} found")
        current_copy = copy.deepcopy(self)
        current_copy.coefficients[key] = item

        return current_copy

    def __iter__(self):
        for coefficient in self.coefficients:
            yield coefficient

    def len(self, bound: float = -numpy.inf, abs_v: bool = False) -> int:
        """
        Length of the premise as number of coefficients above `bound`. If `abs = True`, then the absolute values
        of the coefficients are considered.
        Args:
            bound: Lower bound to include a premise coefficient in the length. Defaults to -numpy.inf.
            abs_v: True to consider the absolute value of coefficients, False otherwise. Defaults to False.

        Returns:
            The length of the premise.
        """
        if abs_v:
            return (abs(self.coefficients) >= bound).sum()
        else:
            return (self.coefficients >= bound).sum()

    def __call__(self, data, **kwargs) -> numpy.ndarray:
        """
        Check whether the given array is within the premise, i.e., whether the premise covers the array or not.
        Args:
            data: The data to check.
            **kwargs:

        Returns:
            An array of coverage where the i-th entry is True if data[i] lies within this hyperplane, False otherwise.
        """
        if not isinstance(data, numpy.ndarray):
            raise ValueError(f"Not a numpy.ndarray: {type(data)}")
        if (data.ndim == 2 and data.shape[1] != self.coefficients.size)\
                or data.ndim == 1 and data.shape[0] != self.coefficients.size:
            raise ValueError(f"Wrong dimensionality: seen {data.shape}, expected ({self.coefficients.shape[0]},)")
        if data.ndim == 2:
            return numpy.array([self(x_) for x_ in data])

        return numpy.dot(self.coefficients, data.transpose()) <= self.bound

    def __copy__(self):
        return Hyperplane(copy.copy(self.coefficients), self.bound)

    def __deepcopy__(self, memodict):
        return Hyperplane(copy.deepcopy(self.coefficients), self.bound)

    def __add__(self, other: Hyperplane) -> Hyperplane:
        """
        Sum this OHyperplane to the other.
        Args:
            other: The hyperplane to sum.

        Returns:
            An hyperplane whose coefficients are the sum of the coefficients, and the bound is the sum of the bounds.
        """
        if isinstance(other, Hyperplane):
            if self.coefficients.size != other.coefficients.size:
                raise ValueError("Expected size {self.coefficients.size}, {other.coefficients.size} found")
            return Hyperplane(self.coefficients + other.coefficients, self.bound + other.bound)
        elif isinstance(other, (int, float)):
            return Hyperplane(self.coefficients + other, self.bound + other)
        else:
            raise ValueError("Not an hyperplane")

    def __sub__(self, other: Union[Tuple[numpy.ndarray, float], Hyperplane]) -> Hyperplane:
        """
        Subtract the other OHyperplane from this.
        Args:
            other: The hyperplane to subtract.

        Returns:
            An hyperplane whose coefficients are the difference of the coefficients, and the bound is the difference
            of the bounds.
        """
        if isinstance(other, Hyperplane):
            if self.coefficients.size != other.coefficients.size:
                raise ValueError("Expected size {self.coefficients.size}, {other.coefficients.size} found")
            return Hyperplane(self.coefficients - other.coefficients, self.bound - other.bound)
        elif isinstance(other, (int, float)):
            return Hyperplane(self.coefficients - other, self.bound - other)
        else:
            raise ValueError("Not an hyperplane")

    def __mul__(self, other):
        """
        Multiply this OHyperplane by the other, if an OHyperplane, or scale its factors, if a scalar..
        Args:
            other: The hyperplane to multiply, or the scaling factor. Can be a tuple (coefficients, bound), an
                    OHyperplane, or a float.

        Returns:
            An hyperplane whose coefficients are the product of the coefficients, and the bound is the product
            of the bounds. If a scalar is given, it multiplies the coefficients and bound by the scalar.
        """
        if isinstance(other, Hyperplane):
            if self.coefficients.size != other.coefficients.size:
                raise ValueError("Expected size {self.coefficients.size}, {other.coefficients.size} found")
            return Hyperplane(numpy.multiply(self.coefficients, other.coefficients), self.bound * other.bound)
        elif isinstance(other, (int, float)):
            return Hyperplane(self.coefficients * other, self.bound * other)
        else:
            raise ValueError("Not an hyperplane")

    def __invert__(self):
        """
        Negate this hyperplane by flipping its sign.
        Returns:
            A OHyperplane with negated coefficients and bound.
        """
        return Hyperplane(-self.coefficients, -self.bound)

    def __neg__(self):
        """
        Invert this hyperplane by flipping its sign.
        Returns:
            A OHyperplane with negated coefficients and bound.
        """
        return ~self

    def json(self) -> List:
        return [self.coefficients.tolist(), self.bound]

    @staticmethod
    def from_json(json_obj) -> Hyperplane:
        return Hyperplane(*json_obj)
