# Hyperplanes & Systems of Hyperplanes
`Hyperplane` and `System` implement n-dimensional (systems of) inequalities.
For symbolic inequalities and systems check out the `learner` module.

Both classes implement standard dunder modules, that is, both `Hyperplane` and `System` can be:
- copied (`copy.copy(hyperplane_1)`);
- deep-copied (`copy.deepcopy(hyperplane_1)`);
- checked for equality (`hyperplane_1 == hyperplane_2`);
- accessed through getters (`hyperplane[i]` or `system[i]`);
- modified through setters (`hyperplane[i] = 0.` or `system[i] = hyperplane`, both **stateless**, that is, they 
  return a copy of the modified object).

They also provide a `json()` method for encoding into a JSON dictionary, and a static method `from_json` for decoding.

Additionally, both objects provide a `Callable` interface: invoking either `hyperplane(x)` or `system(x)` will return
a boolean `numpy.array` such that the i-th entry is `True` if `x[i]` lies within the hyperplane/system, and `False`
otherwise.
```python
import numpy

from hyperplanes.planes import Hyperplane
from systems import System

hyperplane_1 = Hyperplane(numpy.array([1., -0.5, 0.]), 10.)  # 1*x1 -0.5*x2 <= 10.
x = numpy.array([[5., 2., 1.],      # 5*1 -0.5*2 <= 10
                 [22., 20., 0.]])   # 22*1 -0.5*21 <= 10
print(hyperplane_1(x))

hyperplane_1 = Hyperplane(numpy.array([1., -0.5, 0.]), 10.)     # 1*x1 -0.5*x2 <= 10.
hyperplane_2 = Hyperplane(numpy.array([1., 0., 1.]), 5.)        # x1 + x2 <= 5.
system = System([hyperplane_1, hyperplane_2])
print(system(x))
```

## Hyperplanes
The `Hyperplane` class implements a standard Hyperplane of the form `ax <= b`, where `a` are the `coefficients`, and
`b` is the `bound`.
```python
import numpy

from hyperplanes.planes import Hyperplane

dimensionality = 10
hyperplane_1 = Hyperplane(numpy.random.rand(dimensionality,), numpy.random.rand(1,).item())
hyperplane_2 = Hyperplane(numpy.random.rand(dimensionality,), numpy.random.rand(1,).item())
```
`Hyperplanes`s support basic ring operations, which are defined **only on hyperplanes of the same size**.
Applying them to `Hyperplanes`s of different sizes will raise an exception.

**All operations are stateless**, that is, they **do not** modify the original `Hyperplane`, rather they return a copy
on which the operation is performed.
```python
sum_hyperplanes = hyperplane_1 + hyperplane_2   # sums coefficients and bounds
sub_hyperplanes = hyperplane_1 - hyperplane_2   # subtracts coefficients and bounds
mul_hyperplanes = hyperplane_1 * hyperplane_2   # multiplies coefficients and bounds
inverted_hyperplane = ~hyperplane_1             # negates coefficients and bounds
negated_hyperplane = -hyperplane_1              # negates coefficients and bounds
```

## Systems
The `System` class implements a system of inequalities.
```python
import numpy

from hyperplanes.planes import Hyperplane

size = (10, 5)
nr_hyperplanes = 5
hyperplanes = [(numpy.random.rand(size[1],), numpy.random.rand(1,).item()) for _ in range(nr_hyperplanes * 2)]
hyperplanes_1 = [Hyperplane(coeffs, bound) for coeffs, bound in hyperplanes[:nr_hyperplanes]]
hyperplanes_2 = [Hyperplane(coeffs, bound) for coeffs, bound in hyperplanes[nr_hyperplanes:]]

system_1 = System(hyperplanes_1)
system_2 = System(hyperplanes_2)
```

`System`s support basic ring operations, which are defined **only on systems of the same size**.
Applying them to `System`s of different sizes will raise an exception.

**All operations are stateless**, that is, they **do not** modify the original `System`, rather they return a copy
on which the operation is performed.

### Negation [unary]
Negation returns a new `System` with flipped signs on both `coefficients` and `bounds`. 
```python
negated_system_1 = -system_1 
```

### Inversion [unary]
Inversion returns a new `System` with flipped signs on both `coefficients` and `bounds`.
```python
inverted_system_1 = ~system_1 
```

### Addition [binary]
```python
system_add = system1 + system2
```
Addition creates a new System whose coefficients and bounds are the sum of the given system's coefficients and
bounds, respectively.

### Subtraction [binary]
```python
system_sub = system1 - system2
```
Specular to addition, subtraction creates a new System whose coefficients and bounds are the difference of the given 
system's coefficients and bounds, respectively.

### Multiplication [binary]
```python
system_sub = system1 * system2
```
Specular to addition, subtraction creates a new System whose coefficients and bounds are the difference of the given 
system's coefficients and bounds, respectively.
