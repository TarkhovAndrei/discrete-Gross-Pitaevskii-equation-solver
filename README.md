# A solver for the Gross-Pitaevskii equation on a one-, two- or three- dimensional lattice

A tool for modelling the system of coupled Bose-Einstein condensates on a lattice with periodic boundary conditions in 1D, 2D and 3D.

In addition, the code includes:

1) The standard routine for calculating the largest Lyapunov exponent.

2) A custom imperfect time-reversal routine ([Loschmidt echo](http://www.scholarpedia.org/article/Loschmidt_echo)).

3) A custom routine for estimating the typical temperature on a chosen energy shell from conservative dynamics. 

4) Non-conservative quenching terms;
    1) Driving the system to a pre-set energy.
    2) Driving the system according to a pre-set energy-drain profile.

5) Disorder of several kinds:
    1) A local symmetry breaking disorder field (linear in \psi). 
    2) Disorder in the local chemical potential (preserves symmetry, proportional to \asb(\psi^2)).
    3) Disorder in the on-site interaction term (preserves symmetry, proportional to \asb(\psi^4)).

## Numerical integration algorithms used

The dopri45 Runge-Kutta with adaptive time-step (parallelized on CPUs and GPUs versions) support only double precision numbers, and a custom 4-th order Runge-Kutta algorithm with a fixed time step are employed. The custom code supports quadrupole precision floats for exact Lyapunov exponents calculations.

## CPU Parallelization

The code supports OpenMP parallelization on multicore CPUs via [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) solver.

## CPU Parallelization

The code supports GPU parallelization on NVIDIA GPUs via PyTorch + [torchdiffeq](https://github.com/rtqichen/torchdiffeq) solver.

## For citation

The code was used for obtaining numerical results for the papers:

1) [Andrei E. Tarkhov, Sandro Wimberger, and Boris V. Fine, _Phys. Rev. A_ **96**, 023624 (2017)](https://doi.org/10.1103/PhysRevA.96.023624)

2) [Andrei E. Tarkhov, Boris V. Fine, _New J. Phys._ **20** 123021 (2018)](https://doi.org/10.1088/1367-2630/aaf0b6)

-----------------------------------------
Copyright <2019> <Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
https://github.com/TarkhovAndrei/DGPE>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following 2 conditions:

1) If any part of the present source code is used for any purposes followed by publication of obtained results,
the citation of the present code shall be provided according to the rule:

    "Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
    source code from the GitHub repository https://github.com/TarkhovAndrei/DGPE
    was used to obtain the presented results, 2019."

2) The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.