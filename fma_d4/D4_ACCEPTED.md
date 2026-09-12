# Accepted D4 F=ma practice problem

## Problem

On a frictionless horizontal floor sits a movable right wedge of mass `m/2`. Its 45-degree face rises
to the right. A solid sphere of radius `R` and total mass `m` is released from rest on the face; it rolls
downward to the left while the wedge accelerates right. The sphere is spherically symmetric with density
`rho(r)=kappa r`, where `r` is distance from its center. Rolling occurs without slipping relative to the
moving wedge.

The only contact friction is between sphere and wedge. A thin spherical shell of radius `r` has moment
of inertia `(2/3)r^2 dm` about a diameter. Immediately after release, what is the magnitude of the
wedge's horizontal acceleration divided by `g`?

- A. `3/10`
- B. `1/3`
- C. `2/5`
- D. `3/8`
- E. `1/4`

## Answer and generated solution

**A. `3/10`**

Normalize the density:

`m = integral_0^R 4 pi kappa r^3 dr = pi kappa R^4`.

The sphere's moment of inertia is

`I = integral_0^R (2/3)r^2 dm = (4/9)mR^2`,

so `beta=I/(mR^2)=4/9`. Let `s` be displacement down the incline relative to the wedge and `X` the
wedge's rightward displacement. Rolling gives `sdot=R omega`. Horizontal momentum conservation and
the tangential rolling equation give

`Xddot = [m/(m+M_w)] sddot cos(theta)`,

`sddot[(1+beta)-m cos^2(theta)/(m+M_w)] = g sin(theta)`.

For `M_w=m/2` and `theta=45 degrees`, the bracket is

`13/9-(2/3)(1/2)=10/9`.

Therefore `sddot=9g/(10 sqrt(2))`, and

`Xddot/g = (2/3)(1/sqrt(2))(9/(10 sqrt(2))) = 3/10`.

## Independent validation

The blind solver saw only the stem and choices and independently used wedge-frame forces rather than
the generator's momentum/Lagrange derivation. It selected A and reported:

- physics valid: yes
- unique answer: yes
- difficulty: D4
- conceptual deductions: 5
- non-obvious decisions: 1
- familiar template: no
- issues: none

The separate source-aware novelty check reported closest source `NONE`, with different apparatus,
geometry, event sequence, and target form from every indexed source.

## Why this is D4

The solver must normalize a nonuniform three-dimensional density, integrate the rotational inertia,
recognize that rolling is relative to a moving support, exploit horizontal momentum conservation, and
solve coupled translation-rotation dynamics. None of those intermediate quantities is leaked in the stem.
