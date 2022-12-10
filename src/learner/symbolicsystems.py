from __future__ import annotations

import copy
import itertools
import json
import os.path
from abc import abstractmethod, ABC
from typing import Optional, Dict, Sequence, overload, List

import numpy

from hyperplanes.planes import Hyperplane
from hyperplanes.systems import System
from .expressions import Variable, Constant, SymbolicInequality, Operator, GE_Operator, HOVariable

_PARAMETRIZER_PATH = os.path.abspath(__file__)
_PARAMETRIZER_PATH[:_PARAMETRIZER_PATH.rfind("/")] + "/"


class SymbolicSystem(ABC):
    """A generic linear system of inequalities of the form Ax <= b"""
    def ground(self, grounding: Optional[Dict[str, float]] = None) -> SymbolicGroundedSystem | SymbolicParametricSystem:
        """Assign the unbound variables of this system, if any, according to `grounding`."""
        pass


class SymbolicParametricSystem(Sequence[SymbolicInequality], SymbolicSystem):
    """A parametric system generalizing a set of SymbolicSystem (`self.systems`),
    parametrizing each of them with `self.solutions`.

    Attributes:
        inequalities: Sequence of SymbolicInequality.
        solutions: Solutions to self.systems, i.e., list of groundings.
        symbols: Variables in self.inequalities
    """
    def __init__(self, inequalities: Sequence[SymbolicInequality],
                 solutions: Optional[Sequence[Dict[str, float]]] = None):
        self.inequalities = inequalities
        self.solutions = solutions
        symbols = [[copy.deepcopy(symbol) for symbol in inequality if isinstance(symbol, Variable)]
                   for inequality in self.inequalities]
        self.symbols = {symbol.name: symbol for symbol in set(itertools.chain.from_iterable(symbols))}

    @overload
    @abstractmethod
    def __getitem__(self, index: slice) -> Sequence[SymbolicInequality]:
        return self.inequalities[slice.start:slice.stop:slice.step]

    def __getitem__(self, index: int) -> SymbolicInequality:
        return self.inequalities[index]

    def __len__(self):
        return len(self.inequalities)

    def __repr__(self):
        s = "System:\n"
        s += "\n".join(repr(system) for system in self.inequalities)
        s += "\n\t---\nSolutions:\n"
        s += repr(self.solutions)

        return s

    def __str__(self):
        s = "System:\n\t"
        s += "\n\t".join(str(system) for system in self.inequalities)
        s += "\n---\nSolutions:\n"
        for i, grounding in enumerate(self.solutions):
            s += f"\tSolution {i}\n"
            for k, v in grounding.items():
                s += f"\t\t{k} = {v}\n"

        return s

    def rename(self, variable_maps: Dict[str, str]) -> SymbolicParametricSystem:
        """Rename in-place this system's variables according to the given maps.
        Args:
            variable_maps: The map old variable => new variable

        Returns:
            This ParametrizedSystem, with its variables replaced.
        """
        for old_name, new_name in variable_maps.items():
            for i, system in enumerate(self):
                for j, symbol in enumerate(system):
                    if isinstance(symbol, Variable) and symbol.name == old_name:
                        system[j] = symbol.rename(new_name)

        return self

    def ground(self, grounding: Optional[Dict[str, float]] = None) -> SymbolicGroundedSystem | SymbolicParametricSystem:
        """Ground this system, either with the provided `groundings` or this system's solutions.

        Args:
            grounding: The grounding, a dictionary variable name => value

        Returns:
            A (partially, if the grounding dictionary does not contain all the free variables) grounded system.
        """
        inequalities = [SymbolicInequality([symbol.ground(grounding.get(symbol.name, symbol.value))
                                                if isinstance(symbol, Variable) else copy.deepcopy(symbol)
                                            for symbol in equation])
                        for equation in self.inequalities]
        free_variables = ((symbol for symbol in inequality
                           if isinstance(symbol, HOVariable) and symbol.value is None)
                          for inequality in inequalities)
        free_variables = itertools.chain.from_iterable(free_variables)
        free_variable = next(free_variables, None)

        if free_variable is None:
            # fully grounded system
            grounded_system = SymbolicGroundedSystem(inequalities, None)
        else:
            grounded_system = SymbolicParametricSystem(inequalities, copy.deepcopy(self.solutions))

        return grounded_system

    def __hash__(self):
        return sum([hash(system) for system in self.inequalities])

    def __eq__(self, other):
        return isinstance(other, SymbolicParametricSystem)\
                and len(self.inequalities) == len(other.inequalities)\
                and all([this_system == other_system for this_system, other_system in zip(self.inequalities, other.inequalities)])\
                and self.solutions == other.solutions

    def json(self):
        system_obj = [s.json() for s in self.inequalities]
        solutions_obj = self.solutions
        json_obj = {"inequalities": system_obj, "solutions": solutions_obj}

        return json_obj

    @staticmethod
    def from_json(json_obj) -> SymbolicParametricSystem:
        if "inequalities" not in json_obj and "solutions" not in json_obj:
            raise ValueError(f"Not a SymbolicInequality: {json_obj}")
        inequalities = [SymbolicInequality.from_json(system) for system in json_obj["inequalities"]]
        solutions = json_obj["solutions"]

        return SymbolicParametricSystem(inequalities, solutions)


class SymbolicGroundedSystem(SymbolicSystem):
    """A symbolic system with no free variables.

    Attributes:
        inequalities: Sequence of SymbolicInequality defining the system.
        label: A label associated with the system.
    """
    def __init__(self, inequalities: Sequence[SymbolicInequality], label: Optional[float] = None):
        self.inequalities = inequalities
        self.label = label

    def to_system(self, size: int) -> System:
        """Convert this system to a System."""
        hyperplanes = list()
        for equation in self.inequalities:
            # coefficients
            coefficients = numpy.zeros(size,)
            # by construction variables are named X1, X2, ..., hence we can get the reference feature simply
            # by ignoring the X
            reference_variables = list(filter(lambda symbol: isinstance(symbol, Variable)
                                                             and symbol.name.startswith("X"),
                                              equation))
            indexes = [int(symbol.name[1:]) for symbol in reference_variables]
            variables_coefficients = [symbol.coefficient for symbol in reference_variables]
            # bound
            operator_index = [i for i, symbol in enumerate(equation) if isinstance(symbol, Operator)][0]
            inverted_constants = sum([-symbol.value for symbol in equation[:operator_index]
                                      if isinstance(symbol, Constant)])
            constants = sum([symbol.value for symbol in equation[operator_index:] if isinstance(symbol, Constant)])
            bound = constants - inverted_constants

            coefficients[indexes] = variables_coefficients
            hyperplane = Hyperplane(coefficients, bound)

            if isinstance(equation[operator_index], GE_Operator):
                hyperplane = ~hyperplane

            hyperplanes.append(hyperplane)

        system = System(hyperplanes)

        return system

    def ground(self, grounding: Optional[List[Dict[str, float]]] = None) -> List[SymbolicGroundedSystem]:
        return [copy.deepcopy(self)]

    def __deepcopy__(self, memodict):
        return SymbolicGroundedSystem([copy.deepcopy(eq) for eq in self.inequalities])

    def __hash__(self):
        return sum([hash(equation) for equation in self.inequalities])

    def __eq__(self, other):
        return isinstance(other, SymbolicGroundedSystem) \
               and len(self.inequalities) == len(other.inequalities) \
               and all([this_equation == other_equation
                        for this_equation, other_equation in zip(self.inequalities, other.inequalities)])\
               and self.label == other.label

    def json(self):
        system_obj = [s.json() for s in self.inequalities]
        label_obj = self.label
        json_obj = {"inequalities": system_obj, "label": label_obj}

        return json_obj

    @staticmethod
    def from_json(json_obj) -> SymbolicGroundedSystem:
        if "inequalities" not in json_obj and "label" not in json_obj:
            raise ValueError(f"Not a SymbolicInequality: {json_obj}")
        inequalities = [SymbolicInequality.from_json(system) for system in json_obj["inequalities"]]
        label = json.loads(json_obj["label"])

        return SymbolicGroundedSystem(inequalities, label)
