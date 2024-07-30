# MJO-probabilistic-forecast

This repository briefly demonstrate the framework of MJO probabilistic forecast framework based on Liouville's theorem (Liouville 1838). The theorem states that a distribution function is constant while moving along any trajectories in phase space.

Liouville equation is given by
$$
\frac{d \rho}{dt} = \frac{\partial \rho}{\partial t} + \vec{v} \cdot \nabla \rho = -\rho \nabla \cdot \vec{v}
$$
where $\rho$ is ensemble density, $\vec{v}$ is the information flow in the phase space. This equation is mathematically identical to the continuity equation in fluid dynamics. Both of them have similar physical insights, the continuity equation infers the conservation of mass, and Liouville equation infers the conservation of information.

We apply Liouville equation on probabilistic forecast of MJO. The concept can be briefly introduced by the following schematic:

![Concept of applying Liouville equation](Images/schematic.jpg)

## References
Joseph Liouville. Note sur la théorie de la variation des constantes arbitraires. Journal de mathématiques pures et appliquées, 3:342–349, 1838.

## Acknowledgement
Thanks to my advisor, also the original algorithm developer, Dr. Kai-Chi Tseng: https://github.com/kuiper2000