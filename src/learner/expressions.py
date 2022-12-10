"""Module defining symbolic expressions."""

from __future__ import annotations

import copy
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Self, Dict


class SymbolicExpression(ABC):
    def rename(self, new_name: str):
        """Generate a new Expression identical to this one, but with the given `name`.

        Args:
            new_name: The new name

        Returns:
            A new ObliqueExpression with the given `new_name`
        """
        new_expression = copy.deepcopy(self)
        new_expression.name = new_name

        return new_expression

    @staticmethod
    def symbol(string: str | float) -> SymbolicExpression:
        """Generate an ObliqueExpression from the given string"""
        if isinstance(string, float):
            return Constant(value=string)
        if string == "=<":
            return LE_Operator()
        elif string == ">=":
            return GE_Operator()
        elif string == "-":
            return Minus()
        elif string == "+":
            return Plus()
        elif string == "*":
            return Mul()
        if string.isdigit():
            return Constant(value=float(string))
        else:
            if "*" in string:
                coefficient = float(string[:string.index("*")])
                string = string[string.index("*") + 1:]
            else:
                coefficient = 1.0

            return Variable(name=string, coefficient=coefficient, value=None)

    @staticmethod
    def reduce(symbols_chain: List[SymbolicExpression]) -> List[SymbolicExpression]:
        """Simplify the given list of symbols. Invoked upon `from_prolog`.
        Args:
            symbols_chain: The chain to simplify.

        Returns:
            A simplified version of the chain.
        """
        # (coefficient * variable) and (- variable) are recognized as separate "Modifier" expressions by the
        # engine, so need to push the modifiers back into the right-hand symbol, e.g.:
        #   A - 0.25 * E
        # here we need to scan left-to-right and to match  each modifier to the first non-modifier down the
        # sequence
        simple_symbols_chain = list()
        n = len(symbols_chain)
        i = 0
        while i < n:
            current_symbol = symbols_chain[i]

            if isinstance(current_symbol, Constant):
                # can have
                # alpha * X: can't add: Mul will take care of it at the next symbol
                # alpha + X: add and ignore
                # alpha - X: as above
                # alpha <= X: as above
                if i < n - 1:
                    lookahead_at_1 = symbols_chain[i + 1]
                    if isinstance(lookahead_at_1, Mul):
                        i += 1
                    elif isinstance(lookahead_at_1, Plus) \
                            or isinstance(lookahead_at_1, Minus) \
                            or isinstance(lookahead_at_1, Operator):
                        simple_symbols_chain.append(current_symbol)
                        i += 1
                    else:
                        raise ValueError("")
                else:
                    # las element, just add
                    simple_symbols_chain.append(current_symbol)

            elif isinstance(current_symbol, Minus):
                # can have
                # '- Var': invert Var's coefficient and push
                # '- Const': invert Const and repeat
                lookahead_at_1 = symbols_chain[i + 1]
                if isinstance(lookahead_at_1, Constant):
                    lookahead_at_1.value *= -1.0
                    if i + 2 < n:
                        lookahead_at_2 = symbols_chain[i + 2]
                        if isinstance(lookahead_at_2, Plus) or isinstance(lookahead_at_2, Operator):
                            # - alpha + X OR - alpha <= X
                            simple_symbols_chain.append(lookahead_at_1)
                            i += 3
                        elif isinstance(lookahead_at_2, Mul) and isinstance(lookahead_at_2, Variable):
                            # by construction, in '- alpha * X' X can only be a Variable
                            lookahead_at_2 = symbols_chain[i + 2]
                            lookahead_at_2.coefficient = - 1.0 * lookahead_at_1.value

                            simple_symbols_chain.append(lookahead_at_1)
                            i += 4
                        else:
                            raise ValueError("")
                    else:
                        # expression ending with a '- alpha'
                        simple_symbols_chain.append(lookahead_at_1)
                        i += 2
                elif isinstance(lookahead_at_1, Variable):
                    # sub is only negating the variable, flip coefficient and stop
                    lookahead_at_1.coefficient *= -1.0
                    i += 2

            elif isinstance(current_symbol, Mul):
                # can have
                # alpha * Var: update Var's coefficient
                # Var * Var: leave as is
                multiplier = symbols_chain[i - 1]
                lookahead_at_1 = symbols_chain[i + 1]

                if isinstance(multiplier, Constant):
                    # alpha * Var: update Var's coefficient
                    if isinstance(lookahead_at_1, Variable):
                        lookahead_at_1.coefficient *= multiplier.value

                        simple_symbols_chain.append(lookahead_at_1)
                        i += 2
                    else:
                        raise ValueError("")

                elif isinstance(multiplier, Variable):
                    simple_symbols_chain.append(current_symbol)
                    i += 1

            elif isinstance(current_symbol, Plus):
                i += 1

            elif isinstance(current_symbol, Operator)\
                    or isinstance(current_symbol, Variable):
                simple_symbols_chain.append(current_symbol)
                i += 1

            else:
                raise ValueError("")

        return simple_symbols_chain

    @abstractmethod
    def __hash__(self):
        pass

    def ground(self, value) -> Self:
        # implemented as a copy to make it inherit to all symbols
        return copy.deepcopy(self)

    @staticmethod
    def json(obj: SymbolicExpression):
        return obj.__class__._json(obj)

    @staticmethod
    def from_json(json_obj: Dict) -> SymbolicExpression:
        if "symbol" in json_obj:
            return getattr(sys.modules[__name__], json_obj["symbol"]).from_json(json_obj)
        else:
            raise ValueError(f"Not an Expression: {json_obj}")

    @abstractmethod
    def _json(self):
        pass


@dataclass
class Constant(SymbolicExpression):
    value: float

    def __hash__(self):
        return self.value

    def __str__(self):
        return str(self.value)

    def __neg__(self):
        return Constant(-self.value)

    def __copy__(self, memodict):
        return Constant(self.value)

    def __deepcopy__(self, memodict):
        return Constant(self.value)

    def _json(self):
        return {"symbol": "Constant", "value": self.value}

    @staticmethod
    def from_json(json_obj):
        try:
            return Constant(json_obj["value"])
        except KeyError:
            raise ValueError(f"Not a Constant JSON object: f{json_obj}")


@dataclass
class Variable(SymbolicExpression):
    name: str
    coefficient: float
    value: Optional[float] = None

    def __str__(self):
        return f"{self.coefficient}*{self.name}"

    def __hash__(self):
        return hash(f"{self.name, self.coefficient, self.value}")

    def __neg__(self):
        return Variable(name=self.name,
                        coefficient=-self.coefficient,
                        value=-self.value if self.value is not None else None)

    def __copy__(self, memodict):
        return Variable(self.name, self.coefficient, self.value)

    def __deepcopy__(self, memodict):
        return Variable(self.name, self.coefficient, self.value)

    def higher(self) -> HOVariable:
        """Promotes `self` to a HO_Variable.

        Returns:
            A HO_Variable version of this Variable.
        """
        return HOVariable(self.name, self.coefficient, self.value)

    def ground(self, value: float) -> Variable:
        """Generate a new Variable identical to this one, but grounded, i.e., with a set `value`..

        Args:
            value: The grounding value.

        Returns:
            A new Variable with the given `value`
        """
        grounded_variable = copy.deepcopy(self)
        grounded_variable.value = value

        return grounded_variable

    def _json(self):
        return {"symbol": "Variable",
                "name": self.name,
                "coefficient": self.coefficient,
                "value": self.value}

    @staticmethod
    def from_json(json_obj):
        try:
            return Variable(json_obj["name"], json_obj["coefficient"], json_obj["value"])
        except KeyError:
            raise ValueError(f"Not a Variable JSON object: f{json_obj}")


@dataclass
class HOVariable(Variable):
    """Higher-Order variable."""
    name: str
    coefficient: float
    value: Optional[float] = None

    def __neg__(self):
        return HOVariable(name=self.name,
                          coefficient=-self.coefficient,
                          value=-self.value if self.value is not None else None)

    def _json(self):
        return {"symbol": "HO_Variable",
                "name": self.name,
                "coefficient": self.coefficient,
                "value": self.value}

    @staticmethod
    def from_json(json_obj):
        try:
            return HOVariable(json_obj["name"], json_obj["coefficient"], json_obj["value"])
        except KeyError:
            raise ValueError(f"Not a Variable JSON object: f{json_obj}")


##########################################################################################
# Operators (<=, >=)
##########################################################################################
class Operator(SymbolicExpression, ABC):
    pass

    def __invert__(self):
        return copy.deepcopy(self)


@dataclass
class GE_Operator(Operator):
    name: str = "=>"

    def __hash__(self):
        return -1

    def __str__(self):
        return "=>"

    def __deepcopy__(self, memodict):
        return GE_Operator()

    def __invert__(self):
        return LE_Operator()

    def _json(self):
        return {"symbol": "GE_Operator"}

    @staticmethod
    def from_json(json_obj):
        if "symbol" in json_obj and json_obj["symbol"] == "GE_Operator":
            return GE_Operator()
        raise ValueError(f"Not a GE_Operator: {json_obj}")


@dataclass
class LE_Operator(Operator):
    name: str = "=<"

    def __hash__(self):
        return -2

    def __str__(self):
        return "=<"

    def __deepcopy__(self, memodict):
        return LE_Operator()

    def __invert__(self):
        return GE_Operator()

    def _json(self):
        return {"symbol": "LE_Operator"}

    @staticmethod
    def from_json(json_obj):
        if "symbol" in json_obj and json_obj["symbol"] == "LE_Operator":
            return LE_Operator()
        raise ValueError(f"Not a LE_Operator: {json_obj}")


##########################################################################################
# Modifiers apply to variables and either negate them or multiply them by a coefficient
##########################################################################################
@dataclass
class Modifier(SymbolicExpression, ABC):
    """An addition, subtraction, or multiplication modifier."""
    pass


@dataclass
class Plus(Modifier):
    name: str = "+"
    value: float = +1.0

    def __hash__(self):
        return -10

    def __str__(self):
        return "+"

    def __deepcopy__(self, memodict):
        return Plus()

    def _json(self):
        return {"symbol": "Plus"}

    @staticmethod
    def from_json(json_obj):
        if "type" in json_obj and json_obj["type"] == "Plus":
            return Plus()
        raise ValueError(f"Not a Plus: {json_obj}")


@dataclass
class Minus(Modifier):
    name: str = "-"
    value: float = -1.0

    def __hash__(self):
        return -11

    def __str__(self):
        return "-"

    def __deepcopy__(self, memodict):
        return Minus()

    def _json(self):
        return {"symbol": "Minus"}

    @staticmethod
    def from_json(json_obj):
        if "type" in json_obj and json_obj["type"] == "Minus":
            return Minus()
        raise ValueError(f"Not a Minus: {json_obj}")


@dataclass
class Mul(Modifier):
    name: str = "*"

    def __hash__(self):
        return -12

    def __str__(self):
        return "*"

    def __deepcopy__(self, memodict):
        return Mul()

    def _json(self):
        return {"symbol": "Mul"}

    @staticmethod
    def from_json(json_obj):
        if "type" in json_obj and json_obj["type"] == "Mul":
            return Mul()
        raise ValueError(f"Not a Mul: {json_obj}")


@dataclass
class SymbolicInequality(List[SymbolicExpression]):
    equation: List[SymbolicExpression]

    def __eq__(self, other):
        return len(self) == len(other) and all(self_eq == other_eq for self_eq, other_eq in zip(self, other))

    def __hash__(self):
        return sum([hash(self_eq) for self_eq in self])

    def __getitem__(self, item):
        return self.equation[item]

    def __setitem__(self, key, value):
        self.equation[key] = value

    def __len__(self):
        return len(self.equation)

    def __add__(self, other):
        if isinstance(other, list):
            return SymbolicInequality(self.equation + other)
        elif isinstance(other, SymbolicInequality):
            return SymbolicInequality(self.equation + other.equation)
        else:
            raise ValueError("")

    def __iter__(self):
        return (item for item in self.equation)

    def __copy__(self, memodict):
        return SymbolicInequality([copy.copy(symbol) for symbol in self.equation])

    def __deepcopy__(self, memodict):
        return SymbolicInequality([copy.deepcopy(symbol) for symbol in self.equation])

    def __neg__(self):
        return SymbolicInequality([-symbol if isinstance(symbol, (Variable, Constant)) else symbol
                                                         for symbol in self.equation])

    def __invert__(self):
        return SymbolicInequality([~symbol if isinstance(symbol, Operator) else symbol
                                   for symbol in self.equation])

    def __str__(self):
        return " ".join([str(el) for el in self.equation]).replace("1.0*", "")

    def json(self):
        return [SymbolicExpression.json(expr) for expr in self.equation]

    @staticmethod
    def from_json(json_obj):
        return SymbolicInequality([SymbolicExpression.from_json(expr) for expr in json_obj])

    @staticmethod
    def from_hyperplanes(*hyperplanes) -> List[SymbolicInequality]:
        """Convert the given ObliquePath into an ObliqueEquation."""
        equations = list()
        for hyperplane in hyperplanes:
            coefficients, bound = hyperplane.coefficients, hyperplane.bound
            equation = [Variable(name=f"X{i}", coefficient=coefficient, value=None)
                        for i, coefficient in enumerate(coefficients) if coefficient != 0]
            equation.append(LE_Operator())
            equation.append(Constant(bound))
            equations.append(SymbolicInequality(equation))

        return equations

