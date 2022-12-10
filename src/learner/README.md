# Polylearner

## SymbolicExpression and SymbolicInequality
`SymbolicExpression`s are expressions used to compose `SymbolicInequality`s and `SymbolicParametrizedSystem`.
`SymbolicExpression`s are sequences of `Symbol`s:
```
Symbol := Constant | Variable | Operator | Modifier 
```
`Variable`s are defined by a `name`, `coefficient`, and `value` (when grounded).

`SymbolicExpression`s in turn form `SymbolicInequality`, `List`s of `SymbolicExpression` forming inequalities,
which  are at the base of `SymbolicSystem`s.

## SymbolicSystem
A `SymbolicSystem` can either be grounded (`SymbolicGroundedSystem`), that is, without free variables, or not 
grounded (`SymbolicParametricSystem`), that is, it contains free variables.
`SymbolicParametricSystem` can be grounded with the `ground(Optional[List[Dict[str, float]]] = None)` function by 
providing a `List` of grounding dictionaries `Dict[str, float]` that associate to each variable the given  grounding,
yielding a `List[SymbolicGroundedSystem]`. If no grounding is provided, the one in `system.solutions` is used.
Partial groundings are supported too.

```python
from hyperplanes.planes import Hyperplane
from hyperplanes.systems import System
from learner.polylearner import Polylearner

# [-X =< 0, X =< 3, -Y =< 0, Y =< 2]
h1 = Hyperplane([-1., 0.], 0.)
h2 = Hyperplane([1., 0.], 3.)
h3 = Hyperplane([0., -1.], 0.)
h4 = Hyperplane([0., 1.], 2.)
s1 = System([h1, h2, h3, h4])

# [-X =< 0, X =< 2, -Y =< 0, Y =< 3]
h5 = Hyperplane([-1., 0.], 0.)
h6 = Hyperplane([1., 0.], 2.)
h7 = Hyperplane([0., -1.], 0.)
h8 = Hyperplane([0., 1.], 3.)
s2 = System([h5, h6, h7, h8])
```


## Polylearner
Finally, `Polylearner` objects allow us to fit a set of system, learning their generalization, if one exists, and
its solutions.
```python
learner = Polylearner()
learned_system = learner.fit([s1, s2])
print(learned_system)
# System:
#   X1 => 0.0
#   X1 - F =< 0.0
#   X0 + F =< 5.0
#   X0 => 0.0
#   F => 2.0
#   F =< 3.0
# ---
# Solutions:
#   Solution 0
#       F = 2.0
#       E = 3.0
#       D = 0.0
#       C = 0.0
#   Solution 1
#       F = 3.0
#       E = 2.0
#       D = 0.0
#       C = 0.0
```
Each dictionary in `learned_system.solutions` provides a grounding for the given input systems, that is,
`learned_system.ground(learned_system.solutions[0])` gives us the grounding for `s1`.



Note that the system is not in Homogeneous version, but we can convert it by using `system.homo()`.
```python
print(learned_system.homo)
```

`SymbolicSystem`s can be serialized into JSON and retrieved:
```python
with open("example.json", 'w') as log:
    json.dump(learned_system.json(), log)
with open("example.json", 'r') as log:
    res = json.load(log)
    res = SymbolicParametricSystem.from_json(res)

assert learned_system == res
```