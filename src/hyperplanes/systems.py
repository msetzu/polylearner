from __future__ import annotations

import copy
from typing import Optional, Iterable, List

import numpy

from hyperplanes.planes import Hyperplane


class System:
    def __init__(self, hyperplanes: Optional[Iterable[Hyperplane]] = None):
        if hyperplanes is not None:
            self.coefficients = numpy.vstack([h.coefficients for h in hyperplanes])
            self.bounds = numpy.vstack([h.bound for h in hyperplanes]).squeeze()
            self.size = self.coefficients.shape
        else:
            self.coefficients = None
            self.bounds = None
            self.size = (0, 0)

    def __hash__(self):
        return hash(str(self.coefficients) + str(self.bounds))

    def __eq__(self, other):
        return isinstance(other, System)\
            and self.size == other.size\
            and (self.coefficients == other.coefficients).all()\
            and (self.bounds == other.bounds).all()

    def __copy__(self):
        system_copy = System()
        system_copy.coefficients = copy.copy(self.coefficients)
        system_copy.bounds = copy.copy(self.bounds)
        system_copy.size = copy.copy(self.size)

        return system_copy

    def __deepcopy__(self, memodict):
        system_copy = System()
        system_copy.coefficients = copy.deepcopy(self.coefficients)
        system_copy.bounds = copy.deepcopy(self.bounds)
        system_copy.size = copy.deepcopy(self.size)

        return system_copy

    def __len__(self) -> int:
        return self.size[0]

    def __iter__(self):
        return (Hyperplane(self.coefficients[i], self.bounds[i]) for i in range(len(self)))

    def __getitem__(self, item):
        if item > len(self):
            raise IndexError(f"System of size {self.size}, trying to access {item} > {self.size[0]}")
        return Hyperplane(self.coefficients[item], self.bounds[item])

    def __setitem__(self, key: int, value: Hyperplane):
        if key > len(self):
            raise IndexError(f"System of size {self.size}, trying to set {key} > {self.size[0]}")

        current_copy = copy.deepcopy(self)
        current_copy.coefficients[key] = value.coefficients
        current_copy.bounds[key] = value.bound

        return current_copy

    def __neg__(self):
        current_copy = copy.deepcopy(self)
        current_copy.coefficients = - current_copy.coefficients
        current_copy.bounds = - current_copy.bounds

        return current_copy

    def __add__(self, other):
        """If a Hyperplane, add it to this System's equations. If a System, do the same for all hyperplanes."""
        if isinstance(other, System):
            current_copy = copy.deepcopy(self)
            for hyperplane in other:
                current_copy = self + hyperplane

            return current_copy
        elif isinstance(other, Hyperplane):
            current_copy = copy.deepcopy(self)
            current_copy.coefficients = numpy.vstack(self.coefficients, other.coefficients)
            current_copy.bounds = numpy.vstack(self.bounds, other.bound)
            current_copy.size = current_copy.coefficients.shape

            return copy

    def __sub__(self, other):
        """A new system, difference of the self and other."""
        if not isinstance(other, System):
            raise TypeError(f"Not a System: {type(other)}")
        elif self.size != other.size:
            raise ValueError(f"Sizes don't match: {self.size} and {other.size}")
        else:
            current_copy = copy.deepcopy(self)
            current_copy.coefficients = current_copy.coefficients - other.coefficients
            current_copy.bounds = current_copy.bounds - other.bounds

            return current_copy

    def __mul__(self, other):
        """A new system, coefficients and bounds multiplication of each other."""
        if not isinstance(other, System):
            raise TypeError(f"Not a System: {type(other)}")
        elif self.size != other.size:
            raise ValueError(f"Sizes don't match: {self.size} and {other.size}")
        else:
            current_copy = copy.deepcopy(self)
            current_copy.coefficients = current_copy.coefficients * other.coefficients
            current_copy.bounds = current_copy.bounds * other.bounds

            return current_copy

    def __invert__(self):
        return -self

    def __call__(self, data, **kwargs) -> numpy.ndarray:
        """
        Check whether the given array is within the premise, i.e., whether the premise covers the array or not.

        Args:
            data: The data to check.
            **kwargs:

        Returns:
        An array of coverage where the i-th entry is True if data[i] lies within this hyperplane, False otherwise.
        """
        coverages = numpy.array([hyperplane(data) for hyperplane in self])
        coverages = coverages.all(axis=0)

        return coverages

    def __str__(self):
        coefficients_str = "\n\t\t".join(str(c) for c in self.coefficients)
        return f"System:\n\tCoefficients:\n\t\t{coefficients_str}\n\n\tBounds:\n\t\t{str(self.bounds)}"

    def json(self) -> List:
        return [h.json() for h in self]

    @staticmethod
    def from_json(json_obj) -> System:
        return System([Hyperplane.from_json(h) for h in json_obj])
