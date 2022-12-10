from typing import Iterable, Optional

from hyperplanes.systems import System
from learner.prolog.polylearner import PolySolver
from learner.symbolicsystems import SymbolicParametricSystem


class Polylearner:
    """Learns a Symbolic system generalizing the given systems. Generalization is achieved by
    learning a new system with free variables such that different groundings give the original
     systems. Python implementation of https://www.ijcai.org/Proceedings/13/Papers/162.pdf
     based on the original implementation at http://pages.di.unipi.it/ruggieri/software.html """
    def __init__(self):
        self.systems = ["Learned", "Base", "CHull", "Ms", "Cone"]
        self.engine = PolySolver()

    def fit(self, systems: Iterable[System] | str) -> Optional[SymbolicParametricSystem]:
        """
        Learn a parametrized system for the given Systems.

        Args:
            systems: The systems to parametrize.

        Returns:
            A SymbolicParametrizedSystem generalizing the given `systems`, if one exists. None otherwise.
        """
        result = self.engine.solve(systems)

        if result is not None:
            parametrization, variable_maps, solutions = result
            parametrized_system = SymbolicParametricSystem(parametrization, solutions).rename(variable_maps)
        else:
            parametrized_system = None

        return parametrized_system
