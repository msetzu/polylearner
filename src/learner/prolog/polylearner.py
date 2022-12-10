"""
Interface between the Prolog solver and python.
"""
from __future__ import annotations

from collections import ChainMap
from typing import List, Dict, Iterable

from swiplserver import PrologMQI

from hyperplanes.systems import System
from ..expressions import SymbolicInequality, SymbolicExpression, Variable


class PolySolver:
    """
    Solves the parametrization problem.
    """
    def __init__(self):
        self.systems = ["Learned", "Base", "CHull", "Ms", "Cone"]

    def solve(self, system: Iterable[System] | str):
        """Learn a parametrized system for the given oblique paths.

        Args:
            system: The systems to parametrize.

        Returns:
            A pair holding the parametrized generalized version of the given `paths`, if one exists,
            and the assignment  . None otherwise.
        """
        # solve parametrization
        if not isinstance(system, str):
            query = PolySolver.__paths_to_system(system)
        else:
            query = system

        with PrologMQI() as mqi:
            with mqi.create_thread() as prolog_thread:
                prolog_thread.query("[lps]")
                results = prolog_thread.query(query)
                if isinstance(results, bool) and not results:
                    return None

        # map Prolog solutions to Python objects
        variable_maps = {original_name: renaming for renaming, original_name in results[0].items()
                         if renaming not in self.systems}
        parametrization = self._extract_parameterized_system(results)
        solutions = self.solutions(results[0], variable_maps)

        return parametrization, variable_maps, solutions

    def _extract_parameterized_system(self, results: Dict) -> List[SymbolicInequality]:
        learned_system = results[0]["Learned"]
        variable_name_mappings = {original: renaming for renaming, original in results[0].items()
                                  if renaming not in self.systems}
        equations = [self._rec_from_prolog(equation, parsed_left=[], parsed_right=[],
                                           first_order_variables=variable_name_mappings)
                     for equation in learned_system]

        return equations

    def _rec_from_prolog(self, expression: Dict | str | float, parsed_left: List, parsed_right: List,
                         first_order_variables: Dict[str, str]) -> SymbolicInequality:
        """Extract a SymbolicExpression representation from the given Prolog dictionary.


        Unroll the results of the Prolog engine into a List of SymbolicExpression:
            - each element in the list is either a Variable, an Operator, a Constant or a Mul
            - as long as there are no Operator (<= or >=), elements are assumed to be in a sum
            - no additions nor subtractions are defined in the expression, instead element coefficients are negated if
                a subtraction occurs

        Args:
            expression: The expression to unroll, as provided by the Prolog engine
            first_order_variables: Mapping from new to old variable names: any variable in `expression` not belonging
                                    to this collection will be considered a higher-order variable
        """
        operator_symbol = SymbolicExpression.symbol(expression["functor"])
        left_arg, right_arg = expression["args"]
        left_is_symbol, right_is_symbol = isinstance(left_arg, (str, float)), isinstance(right_arg, (str, float))
        
        if left_is_symbol and right_is_symbol:
            # base case
            left_symbol = SymbolicExpression.symbol(left_arg)
            if isinstance(left_symbol, Variable) and left_symbol.name not in first_order_variables:
                left_symbol = left_symbol.higher()
            parsed_left = parsed_left + [left_symbol]

            right_symbol = SymbolicExpression.symbol(right_arg)
            if isinstance(right_symbol, Variable) and right_symbol.name not in first_order_variables:
                right_symbol = right_symbol.higher()

            return SymbolicInequality(parsed_left + [operator_symbol, right_symbol] + parsed_right)
            
        # recurse on the right hand, not on the left
        elif left_is_symbol and not right_is_symbol:
            left_symbol = SymbolicExpression.symbol(left_arg)
            if isinstance(left_symbol, Variable) and left_symbol.name not in first_order_variables:
                left_symbol = left_symbol.higher()
            
            return self._rec_from_prolog(right_arg,
                                         parsed_left=parsed_left + [left_symbol, operator_symbol],
                                         parsed_right=parsed_right,
                                         first_order_variables=first_order_variables)
        
        # recurse on the left hand, not on the right
        elif not left_is_symbol and right_is_symbol:
            right_symbol = SymbolicExpression.symbol(right_arg)
            if isinstance(right_symbol, Variable) and right_symbol.name not in first_order_variables:
                right_symbol = right_symbol.higher()
            
            return self._rec_from_prolog(left_arg,
                                         parsed_left=parsed_left,
                                         parsed_right=[operator_symbol, right_symbol] + parsed_right,
                                         first_order_variables=first_order_variables)
        # recurse on both
        else:
            return self._rec_from_prolog(right_arg,
                                         parsed_left=parsed_left,
                                         parsed_right=parsed_right,
                                         first_order_variables=first_order_variables)\
                    + [operator_symbol]\
                    + self._rec_from_prolog(right_arg,
                                            parsed_left=parsed_left,
                                            parsed_right=parsed_right,
                                            first_order_variables=first_order_variables)

    def solutions(self, prolog_results: Dict, variable_maps: Dict[str, str]) -> List[Dict[str, float]]:
        """Extract the solutions from the Prolog computation.
        Args:
            prolog_results: The results from the Prolog engine.
            variable_maps: Mappings from the variables introduced by Prolog to the original variables.

        Returns:
            A list of solutions in the form of parameter assignments for each system.
        """
        # extract parametrization values
        parameter_values = prolog_results["Ms"]
        parameter_assignments = [{parameter_assignment["args"][1]: parameter_assignment["args"][0]
                                  for parameter_assignment in system_assignment}
                                 for system_assignment in parameter_values]
        # update names w/ variable maps
        parameter_assignments = [[{variable_maps.get(variable, variable): value}
                                  for variable, value in system_assignments.items()]
                                 for system_assignments in parameter_assignments]
        parameter_assignments = [dict(ChainMap(*assignments)) for assignments in parameter_assignments]

        return parameter_assignments

    @staticmethod
    def __path_to_system(system: System) -> str:
        """
        Convert the given path into a string representation.
        Args:
            system: The system to convert

        Returns:
            A str representation of the given `system`.
        """
        res = ""
        for hyperplane in system:
            for i, coefficient in enumerate(hyperplane):
                if coefficient != 0:
                    # need to adjust exponential notation
                    if "e" in str(coefficient):
                        c = str(coefficient)
                        exponential = int(c[c.index("e") + 2:]) - 1
                        if c[0] == "-":
                            c = "-0." + ("0" * exponential) + c[1] + c[3:c.index("e")]
                        else:
                            c = "0." + ("0" * exponential) + c[0] + c[2:c.index("e")]
                        res += f"{c}*X{i} + "
                    else:
                        res += "{0}*X{1} + ".format(coefficient, i)
            res = res[:-2] + f"=< {hyperplane.bound}, "  # prolog uses =< instead of <=
        res = "[" + res[:-2] + "]"  # remove the dangling ", "
        res = res.replace("+ -", "- ")
        res = res[:-1] if res[-1] == "," else res

        return res

    @staticmethod
    def __paths_to_system(systems: Iterable[System]) -> str:
        """
        Convert the given `systems` into a query to feed to the solver.

        Args:
            systems: The systems to convert

        Returns:
            A query for the Prolog solver.
        """
        systems = [PolySolver.__path_to_system(path) + "," for path in systems]
        systems = "[" + " ".join(systems)[:-1] + "]"  # -1 to remove the trailing ","
        query = f"learning({systems}, Cone, Ms, CHull, Base, Learned)"

        return query
