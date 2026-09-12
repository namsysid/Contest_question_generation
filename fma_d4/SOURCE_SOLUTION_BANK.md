# F=ma 2024-2026 verified source-solution reference bank

These records are retained as readable generator context and are not re-embedded. `blind_answer_agreement` means GPT-5 generated the derivation and GPT-5-mini independently selected the same answer from the question alone. `manual_equation_review` records an explicit equation audit after verifier service failures.

## 2024_fnet_ma_exam_Q09

- Answer: **C**

- Difficulty: **D1**

- Verification: `blind_answer_agreement`

### Source question

When a car’s brakes are fully engaged, it takes 100 m to stop on a dry road, which has coefficient of kinetic
friction µk = 0.8 with the tires. Now suppose only the first 50 m of the road is dry, and the rest is covered
with ice, with µk = 0.2. What total distance does the car need to stop?
(A) 150 m
(B) 200 m
(C) 250 m
(D) 400 m
(E) 850 m 2024

### Structured solution

Frame: along-car motion positive; friction opposes motion (negative). With brakes locked, kinetic friction applies throughout. On fully dry road (μk=0.8) the stopping from speed v0 over 100 m gives v0^2 = 2 μdry g (100) = 160 g. Now first 50 m are dry, so v1^2 = v0^2 − 2 μdry g (50) = 160 g − 80 g = 80 g. On ice (μice=0.2), stopping distance from speed v1 is d2 = v1^2/(2 μice g) = (80 g)/(0.4 g) = 200 m. Total distance = 50 m + 200 m = 250 m. Choice (C).

### Indispensable steps

1. Use v0^2 = 2 μdry g d on the 100 m dry stop to get v0^2 = 160 g
2. After 50 m dry: v1^2 = v0^2 − 2 μdry g (50) = 80 g
3. Ice stopping distance: d2 = v1^2/(2 μice g) = 200 m; add first 50 m

### Independent check

Initial speed v0 satisfies v0^2=2 g µ_k(dry) (100). With mixed surface, stopping distance D obeys v0^2=2 g[µ1(50)+µ2(D-50)]. Equate: 2gµ1(100)=2g[µ1(50)+µ2(D-50)] ⇒ µ1·50=µ2·(D-50). Plug µ1=0.8, µ2=0.2: (0.8)50=0.2(D-50) ⇒ D-50=200 ⇒ D=250 m.

## 2024_fnet_ma_exam_Q23

- Answer: **D**

- Difficulty: **D2**

- Verification: `blind_answer_agreement`

### Source question

Collisions between ping pong balls and paddles are not perfectly elastic. Suppose that if a player holds
a paddle still and drops a ball on top of it from any height h, it will bounce back up to height h/2. To
keep the ball bouncing steadily, the player moves the paddle up and down, so that it is moving upward
with speed 1.0 m/s whenever the ball hits it. What is the height to which the ball is bouncing?
(A) 0.21 m
(B) 0.45 m
(C) 1.0 m
(D) 1.7 m
(E) There is not enough information to determine the height. 2024

### Structured solution

Take up positive. Restitution with a massive paddle moving at speed U gives v+ = (1+e)U − e v−. For a still paddle (U=0), a drop from height h returns to h/2, so e^2 = 1/2 → e = 1/√2. In steady bouncing with the paddle having the same upward speed U at each impact and returning to the same impact height each cycle, the pre- and post-impact speed magnitudes match: |v−| = |v+| = u. With v− = −u, v+ = (1+e)U + e u, so u = (1+e)U + e u ⇒ u(1−e) = (1+e)U ⇒ u = [(1+e)/(1−e)] U. The bounce height is h = u^2/(2g) = [(1+e)^2/(1−e)^2] U^2/(2g). With e = 1/√2 and U = 1.0 m/s, u ≈ 5.83 m/s, so h ≈ 1.7 m. Choice (D).

### Indispensable steps

1. Infer e from stationary bounce: e^2=1/2
2. Use v+ = (1+e)U − e v− with U fixed at impact
3. Steady cycle at same impact height implies |v+|=|v−|; solve u
4. Compute h = u^2/(2g)

### Independent check

Stationary-paddle bounce gives rebound height h' = e^2 h = h/2 ⇒ e^2=1/2 ⇒ e=1/√2. Let s=√(2gH) be impact speed from height H (downward). Paddle speed u upward=1.0 m/s. In paddle frame v_rel = -s - u; after collision v'_lab = u - e(v_rel) = e s + (1+e)u. For steady bounce we require upward v' = s ⇒ s = e s + (1+e)u ⇒ s(1-e)=(1+e)u ⇒ s = ((1+e)/(1-e)) u. With e=1/√2 and u=1 m/s compute H=s^2/(2g) ≈1.7 m.

## 2024_fnet_ma_exam_Q24

- Answer: **B**

- Difficulty: **D2**

- Verification: `blind_answer_agreement`

### Source question

When a projectile falls through a fluid, it experiences a drag force proportional to the product of its
cross-sectional area, the fluid density ρf, and the square of its speed. Suppose a sphere of density ρs ≫ρf
of radius R is dropped in the fluid from rest. When the projectile has reached half of its terminal velocity,
which of the following is its displacement proportional to?
(A) R p ρs/ρf
(B) R ρs/ρf
(C) R (ρs/ρf)3/2
(D) R (ρs/ρf)2
(E) R (ρs/ρf)3

### Structured solution

Downward positive. Quadratic drag: m dv/dt = mg − k v^2 with k ∝ ρf A and A ∝ R^2. Terminal speed satisfies mg = k v_t^2 ⇒ v_t^2 ∝ (ρs R^3 g)/(ρf R^2) = (ρs/ρf) R g. From rest, solution is v(t) = v_t tanh(gt/v_t) and y(t) = (v_t^2/g) ln cosh(gt/v_t). At v = v_t/2, set tanhξ = 1/2 ⇒ ξ = atanh(1/2), so y = (v_t^2/g) ln[coshξ] = (v_t^2/g) ln(2/√3). Hence displacement ∝ v_t^2/g ∝ (ρs/ρf) R. Choice (B).

### Indispensable steps

1. Set mg = k v_t^2 to get v_t^2 ∝ (ρs/ρf) R g
2. Use v = v_t tanh(gt/v_t), y = (v_t^2/g) ln cosh(gt/v_t) from the ODE
3. At v = v_t/2, y ∝ v_t^2/g ⇒ ∝ (ρs/ρf) R

### Independent check

Drag ∼ k v^2 with k∝ρ_f A. Terminal speed v_t satisfies k v_t^2 ∼ m g ⇒ v_t^2 ∼ (m g)/k ∼ (ρ_s R^3 g)/(ρ_f R^2)= (ρ_s/ρ_f) g R. The distance to reach a fraction of v_t scales as v_t^2/g ∼ (ρ_s/ρ_f) R. Hence displacement ∝ R (ρ_s/ρ_f).

## 2025_fnet_ma_exam_Q16

- Answer: **A**

- Difficulty: **D1**

- Verification: `blind_answer_agreement`

### Source question

Two soap bubbles of radii R1 = 1 cm and R2 = 2 cm conjoin together in the air, such that a narrow bridge
forms between them. Assuming the system starts in equilibrium, the bubbles are extremely thin, and that
air can flow freely between the bubbles through the bridge, describe the evolution and final state of the
bubbles.
(A) The smaller bubble will shrink and the larger bubble will grow.
(B) The larger bubble will shrink and the smaller bubble will grow.
(C) The bubbles will maintain their sizes.
(D) Air will oscillate between the two bubbles.
(E) Both bubbles will simultaneously shrink. 2025

### Structured solution

Frame: ambient air at pressure P0; bubbles are thin soap films with surface tension γ. For a soap bubble, the Laplace overpressure is ΔP = P_in − P0 = 4γ/R (two surfaces). Initially R1 = 1 cm, R2 = 2 cm, so ΔP1 = 4γ/R1 > ΔP2 = 4γ/R2, hence P1 > P2. Once a bridge forms allowing flow, air moves from higher to lower pressure (from the smaller to the larger bubble). As R1 decreases, ΔP1 increases; as R2 increases, ΔP2 decreases. Equal-pressure equilibrium requires R1 = R2, which is not satisfied and is unstable for unequal radii. Therefore the smaller bubble continually shrinks and the larger grows until the smaller vanishes, leaving a single larger bubble. Choice (A).

### Indispensable steps

1. Use ΔP = 4γ/R for bubbles; smaller radius implies higher internal pressure.
2. Air flows from high to low pressure; flow direction is small → large.
3. Only equal radii give equal pressure; unequal radii drive runaway in same direction.
4. Conclusion: small shrinks, large grows until small disappears.

### Independent check

Pressure inside a thin soap bubble = P_out + 4γ/R (two surfaces). Smaller bubble (R1=1 cm) has larger internal pressure than larger (R2=2 cm), so air flows from small to large until the small bubble shrinks (eventually disappears) and the large bubble grows. Thus the smaller shrinks and the larger grows.

## 2026_fnet_ma_exam_Q09

- Answer: **B**

- Difficulty: **D3**

- Verification: `blind_answer_agreement`

### Source question

A vertical cylindrical pipe with a sealed bottom contains no air and is tall enough that its top is open
to outer space (no atmosphere). A heavy plate is mounted at the bottom of the cylinder and vibrates
vertically. A small ball inside the pipe undergoes elastic collisions with the plate.

The plate vibrates in such a way that, during each collision, it is moving upward with a constant speed
equal to
v

1000, where v is the orbital speed at zero altitude above the Earth. The plate moves downward
sufficiently fast that the ball collides with the plate only while the plate is moving upward. The ball is
initially dropped from rest from a height equal to the Earth’s radius above the plate.

After how many collisions with the plate will the ball escape the Earth’s gravitational field?
(A) 207
(B) 208
(C) 414
(D) 415
(E) 1000 2026

### Structured solution

Frame: inertial Earth-centered; take upward +x. Central gravity implies the speed at a given radius is the same on ascent and return. The ball is dropped from rest at r0 = 2RE to r = RE, so its first pre-impact speed is s0 = sqrt(2GM(1/RE − 1/(2RE))) = sqrt(GM/RE) ≡ v (orbital speed at zero altitude). An elastic collision with an upward-moving massive plate of speed u adds 2u to the ball’s speed each bounce: if pre-impact velocity is −s (downward), post-impact is +s′ with s′ = 2u + s. Between collisions it returns to the plate with the same speed magnitude. Hence after n collisions the upward speed is s_n = v + 2nu, with u = v/1000. Escape from r = RE requires s_n ≥ v_e = sqrt(2) v. Thus 1 + 0.002 n ≥ sqrt(2) ⇒ n ≥ (sqrt(2) − 1)/0.002 ≈ 207.106…, so the smallest integer is n = 208. Choice (B).

### Indispensable steps

1. Compute first impact speed from 2-body energy: s0 = v.
2. Elastic collision with plate at speed u adds 2u per bounce: s_n = v + 2nu.
3. Escape threshold at surface: s_n ≥ sqrt(2) v.
4. Solve for n with u = v/1000 ⇒ n = 208.

### Independent check

Let v_orb = orbital speed at surface; plate speed u = v_orb/1000. Drop from rest from r=2R to r=R gives impact speed s1 with 1/2 s1^2 = GM/(2R) => s1 = v_orb. For an elastic collision with a massive plate moving up at u, incoming speed s (downward) becomes outgoing upward speed s+2u; thus approach speed increases by 2u each collision. After n collisions the post-collision upward speed is v_after = s1 + 2u n = v_orb + 2u n. Escape requires v_after ≥ v_esc = √2 v_orb ⇒ 2u n ≥ v_orb(√2−1) ⇒ n ≥ [v_orb(√2−1)]/(2u) = 500(√2−1) ≈ 207.11 ⇒ minimal integer n = 208.

## 2026_fnet_ma_exam_Q10

- Answer: **C**

- Difficulty: **D4**

- Verification: `blind_answer_agreement`

### Source question

Consider the following three solid objects, each of mass M and uniform density:
(i) a half-ball of radius a;
(ii) a cylinder of radius a and height a;
(iii) a cylinder of radius a and height 2a.

For each object, consider the gravitational acceleration at the center of its flat circular face. Let gball, ga,
and g2a denote these accelerations for the half-ball, the cylinder of height a, and the cylinder of height
2a, respectively. Which of the following is correct?
(A) ga > gball > g2a
(B) gball > g2a > ga
(C) gball > ga > g2a
(D) g2a > gball > ga
(E) g2a > ga > gball 2026

### Structured solution

Let z-axis be the symmetry axis, pointing into the object from the face center; report magnitudes. Densities (all mass M): ρ_hemi = 3M/(2π a^3), ρ_a = M/(π a^3), ρ_2a = M/(2π a^3). Field on axis at the center of a cylinder’s face (height H) from disk-stacking: g_cyl(H) = 2π G ρ [a + H − sqrt(H^2 + a^2)]. Thus ga = 2π G ρ_a [2a − a√2] = (GM/a^2)[2(2 − √2)] ≈ 1.1716 GM/a^2, and g2a = 2π G ρ_2a [3a − a√5] = (GM/a^2)(3 − √5) ≈ 0.7639 GM/a^2. For the hemisphere, stack disks of radius r(z) = √(a^2 − z^2), at distance z ∈ [0,a]; a disk contributes dg = 2π G ρ dz [1 − z/√(z^2 + r^2)] = 2π G ρ (1 − z/a) dz, so gball = ∫_0^a 2π G ρ_hemi (1 − z/a) dz = π G ρ_hemi a = (3/2) GM/a^2 = 1.5 GM/a^2. Therefore gball > ga > g2a. Choice (C).

### Indispensable steps

1. Express densities in terms of M and geometry.
2. Use axial disk-field formula to integrate cylinders: g_cyl(H) = 2πGρ[a+H−√(H^2+a^2)].
3. Integrate hemisphere as variable-radius disks: g_hemi = π G ρ_hemi a.
4. Compare numerical coefficients: 1.5 > 1.1716 > 0.7639.

### Independent check

Compute axial field at flat-face center for each body (mass M, radius a).
1) Hemisphere: field g_ball = π G ρ a, mass M = (2/3)π a^3 ρ ⇒ ρ = 3M/(2π a^3) ⇒ g_ball = (3/2) G M / a^2 ≈1.5 (GM/a^2).
2) Cylinder height a: general cylinder (radius a, height h) gives g = (2GM/(a^2 h))(h + a − √(a^2+h^2)). For h=a ⇒ g_a = (2GM/a^2)(2−√2) ≈1.1716 (GM/a^2).
3) Cylinder height 2a ⇒ g_2a = (GM/a^2)(3−√5) ≈0.7639 (GM/a^2).
Ordering: g_ball (1.5) > g_a (1.1716) > g_2a (0.7639). Choice C.

## 2024_fnet_ma_exam_Q01

- Answer: **B**

- Difficulty: **D2**

- Verification: `blind_answer_agreement`

### Source question

An archer fires an arrow from the ground so that it passes through two hoops, which are both a height h
above the ground. The arrow passes through the first hoop one second after the arrow is launched, and
through the second hoop another second later. What is the value of h?
(A) 5 m
(B) 10 m
(C) 12 m
(D) 15 m
(E) There is not enough information to decide.

### Structured solution

Frame: inertial Earth frame. Take upward +y, origin at ground. Arrow launched from y=0 with vertical component v0y. Vertical motion: y(t)=v0y t - (1/2) g t^2. Given y(1)=h and y(2)=h: h=v0y - g/2 and h=2 v0y - 2 g. Equate: v0y - g/2 = 2 v0y - 2 g ⇒ v0y = (3/2) g. Then h = v0y - g/2 = g ≈ 9.8 m ≈ 10 m. Choice mapping: (B).

### Indispensable steps

1. Write y(t)=v0y t - (1/2) g t^2 with y(1)=h and y(2)=h
2. Solve the two equations for v0y, then for h
3. Evaluate h = g ≈ 10 m

### Independent check

Take vertical up positive. y(t)=v0y t - (1/2) g t^2. At t=1 and t=2 both equal h: v0y - (1/2)g = h and 2v0y - 2g = h. Equate: 2v0y-2g = v0y - (1/2)g => v0y = (3/2)g. Then h = v0y - (1/2)g = g ≈9.8 m ≈10 m → (B).

## 2024_fnet_ma_exam_Q02

- Answer: **A**

- Difficulty: **D3**

- Verification: `blind_answer_agreement`

### Source question

An amusement park ride consists of a circular, horizontal room. A rider leans against its frictionless outer
walls, which are angled back at 30◦with respect to the vertical, so that the rider’s center of mass is 5.0 m
from the center of the room. When the room begins to spin about its center, at what angular velocity
will the rider’s feet first lift off the floor?
(A) 1.9 rad/s
(B) 2.3 rad/s
(C) 3.5 rad/s
(D) 4.0 rad/s
(E) 5.6 rad/s

### Structured solution

Frame: inertial lab frame. Threshold: feet lift when floor normal Nf=0. The frictionless wall’s normal N is tilted α=30° above horizontal (since wall is 30° from vertical). Force balance: radial (toward center) N cosα = m ω^2 r; vertical N sinα − mg = 0. Eliminate N: ω^2 = (g cotα)/r. With α=30° (cot30°=√3) and r=5.0 m: ω = sqrt((9.8×√3)/5) ≈ 1.84 rad/s ≈ 1.9 rad/s. Choice mapping: (A).

### Indispensable steps

1. Identify threshold Nf=0 and decompose frictionless wall normal at 30° above horizontal
2. Write radial N cosα = m ω^2 r and vertical N sinα = mg
3. Solve ω^2 = g cotα / r and evaluate for α=30°, r=5 m

### Independent check

Choose axes: vertical up positive, radial inward positive. Wall is 30° from vertical ⇒ normal N points 30° above horizontal. At lift-off floor normal = 0 so vertical balance: N sin30 = mg ⇒ N=2mg. Radial (centripetal): N cos30 = m ω^2 r ⇒ ω^2 = (N cos30)/(m r) = (2g cos30)/r = (g√3)/r. With r=5.0 m and g≈9.8 ⇒ ω = sqrt(g√3/5) ≈1.84 rad/s ≈1.9 rad/s → (A).

## 2024_fnet_ma_exam_Q22

- Answer: **D**

- Difficulty: **D1**

- Verification: `blind_answer_agreement`

### Source question

A spherical shell is made from a thin sheet of material with a mass per area of σ. Consider two points,
P1 and P2, which are close to each other, but just inside and outside the sphere, respectively. If the
accelerations due to gravity at these points are g1 and g2, respectively, what is the value of |g1 −g2|?
(A) πGσ
(B) 4πGσ/3
(C) 2πGσ
(D) 4πGσ
(E) 8πGσ

### Structured solution

Choose outward radial as +r. A thin spherical shell of surface mass density σ has g_out = -GM/R^2 r̂ with M=4πR^2σ, so g_out = -4πGσ r̂ (inward). Inside a shell, g_in = 0. Hence the jump of the normal component is (g_out - g_in)·(+r̂) = -4πGσ, and the magnitude difference is |g1 - g2| = |0 - (-4πGσ)| = 4πGσ. Maps to choice (D).

### Indispensable steps

1. Use shell theorem/Gauss’s law: inside field of a spherical shell is zero
2. Outside field equals that of a point mass M=4πR^2σ at center: |g_out|=4πGσ
3. Take the difference across the surface to get |g1−g2|=4πGσ

### Independent check

Shell mass M=4πR^2σ. By shell theorem g_inside=0; just outside g=GM/R^2=G(4πR^2σ)/R^2=4πGσ. Hence |g1−g2|=4πGσ.

## 2025_fnet_ma_exam_Q08

- Answer: **C**

- Difficulty: **D1**

- Verification: `blind_answer_agreement`

### Source question

A symmetric spinning top, rotating clockwise at an angular frequency ω, is placed upright in the center
of a frictionless circular plate. The plate then begins to rotate counterclockwise at a constant angular
velocity ω. Assume the top’s axis remains perfectly vertical and stable without any precession. From the
perspective of an observer rotating with the plate, how does the top appear to rotate?
(A) The top appears stationary without any rotation.
(B) The top appears to rotate in the clockwise direction at an angular frequency ω.
(C) The top appears to rotate in the clockwise direction at an angular frequency 2ω.
(D) The top appears to rotate in the counterclockwise direction at an angular frequency ω.
(E) The top appears to rotate in the counterclockwise direction at an angular frequency 2ω.

### Structured solution

Take +z as CCW. In the lab: Ω_top = -ω ẑ (clockwise), Ω_frame (plate) = +ω ẑ. In the plate’s rotating frame, apparent spin is Ω_rel = Ω_top − Ω_frame = -2ω ẑ, i.e., clockwise at angular frequency 2ω. Choice (C).

### Indispensable steps

1. Define sign: +z is CCW
2. Use rotating-frame relation Ω_rel=Ω_obj−Ω_frame
3. Compute Ω_rel=(−ω)−(+ω)=−2ω ⇒ clockwise at 2ω

### Independent check

Take CCW positive. Top angular velocity in lab ω_top = -ω (clockwise). Plate/frame ω_frame = +ω. Apparent ω = ω_top - ω_frame = -ω - ω = -2ω → clockwise at 2ω (choice C).

## 2025_fnet_ma_exam_Q12

- Answer: **A**

- Difficulty: **D2**

- Verification: `blind_answer_agreement`

### Source question

A 50 g piece of clay is thrown horizontally with a velocity of 20 m/s striking the bob of a stationary
pendulum with length l = 1 m and a bob mass of 200 g. Upon impact, the clay sticks to the pendulum
weight and the pendulum starts to swing. What is the maximum change in angle of the pendulum?
(A) arccos(1/5)
(B) arcsin(7/10)
(C) arccos(2/3)
(D) arcsin(3/10)
(E) arctan(4/5)

### Structured solution

Let x be horizontal along the throw. Masses: m_c=0.05 kg, m_b=0.20 kg, M=0.25 kg, l=1 m. During the short impact at the bottom (string vertical), the pivot force is vertical, so x-momentum is conserved: m_c v0 = M v+ ⇒ v+ = (0.05·20)/0.25 = 4 m/s. After sticking, mechanical energy is conserved until the turning point: (1/2)M v+^2 = M g l (1−cosθ). Using g≈10 m/s^2: 8 = 10(1−cosθ) ⇒ cosθ = 1/5 ⇒ θ = arccos(1/5). Choice (A).

### Indispensable steps

1. Conserve horizontal momentum in the collision (pivot impulse is vertical) to get v+
2. Convert post-collision KE to gravitational PE at the turning point to find θ
3. Insert numbers (g≈10) to obtain cosθ=1/5

### Independent check

Conserve horizontal momentum at impact: 0.05·20=(0.2+0.05)v' ⇒ v'=4.0 m/s. KE→PE: ½(0.25)v'^2 = (0.25)g l(1−cosθ) ⇒ cosθ = 1 − v'^2/(2 g l) = 1 − 16/(2 g). Using g≈10 m/s^2 (to match choices) gives cosθ=0.2 ⇒ θ=arccos(1/5) (A).

## 2025_fnet_ma_exam_Q14

- Answer: **D**

- Difficulty: **D3**

- Verification: `blind_answer_agreement`

### Source question

A ball launcher fires balls along the floor at the same initial speed with no initial rotation. Each ball initially slips, then rolls without slipping. Ignore deformation and air resistance. There are three kinds of balls that can be launched in this set-up, all having the same radius R:
I. a regular tennis ball (a thin spherical shell of rubber) of mass m1
II. a solid wooden ball of mass m2
III. a solid rubber ball of mass m3
where m1 < m2 < m3. All three types of ball emerge from the launcher with the same velocity. For which
ball will the final velocity be highest?
(A) Ball I
(B) Ball II
(C) Ball III
(D) Balls II and III
(E) The final velocity will be the same for all three balls

### Structured solution

Let +x be forward. Initially v(0)=v0>0, ω(0)=0. While slipping: kinetic friction fk=μk m g acts at the contact opposite slip ⇒ a=−μk g. Torque about CM: τ= fk R=μk m g R ⇒ α=τ/I=μk m g R/I (spin forward). Thus v(t)=v0−μk g t, ω(t)=μk m g R t/I. Rolling starts when v=ωR ⇒ v0−μk g t* = (μk m g R^2/I)t* ⇒ t* = v0/[μk g (1 + mR^2/I)]. Then v_f = v(t*)= v0·[mR^2/(I+mR^2)]. Evaluate: thin spherical shell I=(2/3)mR^2 ⇒ v_f=(3/5)v0; solid sphere I=(2/5)mR^2 ⇒ v_f=(5/7)v0. Hence solids are faster, and mass cancels, so balls II and III tie. Choice (D).

### Indispensable steps

1. Write translational and rotational equations with kinetic friction sign
2. Impose rolling threshold v=ωR to solve for t*
3. Obtain v_f = v0·[mR^2/(I+mR^2)] and compare I/(mR^2) for each ball

### Independent check

Let I = c m R^2. While slipping: a = -μg, α = μg/(cR). Rolling when v0 - μg t = α R t ⇒ v0 = μg t(1+1/c) ⇒ v_f = v0 - μg t = v0/(1+c). For thin shell c=2/3 ⇒ v_f=3/5 v0; for solid sphere c=2/5 ⇒ v_f=5/7 v0. Solids (II & III) have the same, larger final speed ⇒ (D).

## 2025_fnet_ma_exam_Q17

- Answer: **B**

- Difficulty: **D3**

- Verification: `blind_answer_agreement`

### Source question

A particle of mass m moves in the xy plane with potential energy U(x,y) = (-k x^2 + y^2)/2.

The closest point to the origin (x = 0, y = 0) during its motion was at a distance d, and the particle’s
speed at that point was v ̸= 0. Which of the following statements is true regarding the path of the particle
after a long time t (t ≫d/v)?
(A) The particle’s trajectory will be circular.
(B) The particle’s trajectory will be asymptotic to a straight line pointing away from the origin.
(C) The particle will spiral outwards away from the origin.
(D) The particle will travel on a parabolic trajectory.
(E) The particle will spiral inwards towards the origin.

### Structured solution

Take an inertial xy frame. U(x,y) = (−k x^2 + y^2)/2 gives forces F = −∇U:
Fx = −∂U/∂x = +k x, Fy = −∂U/∂y = − y.
Equations: m ẍ = k x, m ÿ = − y. Let λ = √(k/m), ω = √(1/m). General solutions:
x(t) = A e^{λ t} + B e^{−λ t}, y(t) = C cos(ω t) + D sin(ω t).
At the instant of closest approach r = √(x^2 + y^2) = d with nonzero speed v, we have d(r^2)/dt = 2(x ẋ + y ẏ) = 0 but (ẋ, ẏ) ≠ (0,0).
If A = 0, then x → 0 as t → ∞ while y(t) oscillates through y = 0 infinitely many times, implying the global minimum distance would be 0, contradicting the given finite d > 0. Hence A ≠ 0. For t ≫ d/v, x(t) ~ A e^{λ t} grows unbounded, while y(t) remains bounded. Therefore the trajectory becomes asymptotic to a straight line parallel to the x-axis, with the position vector pointing away from the origin. Choice (B).

### Indispensable steps

1. Compute m ẍ = k x and m ÿ = − y from U
2. Solve to get x = A e^{λ t}+B e^{−λ t}, y bounded sinusoid
3. Use d>0 to exclude A=0 (else later r→0)
4. Conclude x → ±∞, y bounded ⇒ path asymptotic to straight line away from origin

### Independent check

U = (-k x^2 + y^2)/2 => F_x = -∂U/∂x = k x, F_y = -∂U/∂y = -y. Equations: m x'' = k x (unstable, solutions ∝ e^{±√(k/m) t}), m y'' = - y (stable oscillation, ω_y = √(1/m)). For generic nonzero initial condition x grows exponentially while y remains bounded oscillatory, so y/x → 0 and the trajectory becomes asymptotic to a straight line pointing away from the origin. 

## 2025_fnet_ma_exam_Q18

- Answer: **D**

- Difficulty: **D3**

- Verification: `blind_answer_agreement`

### Source question

A particle of mass m moves in the xy plane with potential energy

U(x,y) = kxy/2.

If the particle begins at the origin, then it is possible to displace it slightly in some direction, so that the
particle subsequently oscillates periodically. What is the period of this motion?
(A) 2 pi sqrt(m/(4k))
(B) 2 pi sqrt(m/(2k))
(C) 2 pi sqrt(m/k)
(D) 2 pi sqrt(2m/k)
(E) 2 pi sqrt(4m/k)

### Structured solution

In an inertial frame, U = (k/2) x y. With U = (1/2) q^T K q and symmetric K, choose K = [[0, k/2],[k/2, 0]]. Then m q̈ + K q = 0.
Diagonalize with normal coordinates u = (x + y)/√2 and v = (x − y)/√2. Eigenvalues of K: +k/2 for eigenvector (1,1) (u-mode), and −k/2 for (1,−1) (v-mode). Thus
m ü + (k/2) u = 0 (stable), m v̈ − (k/2) v = 0 (unstable).
Starting at the origin, choose a small displacement purely along the stable eigen-direction x = y (i.e., u ≠ 0, v = 0). Then the motion is simple harmonic with ω = √((k/2)/m), period T = 2π/ω = 2π √(2m/k). Choice (D).

### Indispensable steps

1. Form K with K_xy=K_yx=k/2, K_xx=K_yy=0
2. Rotate to u=(x+y)/√2, v=(x−y)/√2
3. Identify stable u-mode: m ü+(k/2)u=0
4. Period T=2π√(2m/k) for pure u excitation

### Independent check

U = (k/2)xy => m x'' = - (k/2) y, m y'' = - (k/2) x. Assume x,y ∝ e^{iωt}; get m^2 ω^4 = (k/2)^2 so ω^2 = ± k/(2m). The stable oscillatory mode has ω = √(k/(2m)), hence period T = 2π/ω = 2π √(2m/k).

## 2025_fnet_ma_exam_Q19

- Answer: **E**

- Difficulty: **D1**

- Verification: `blind_answer_agreement`

### Source question

Near the ground, wind speed can be modeled as proportional to height above the ground. (This is a
reasonable assumption for small heights.) A wind turbine converts a constant fraction of the available
kinetic energy into electricity. The conditions are such that when operating at 10 m above the ground, the
turbine delivers 15 kW of power. How much power would the same windmill deliver if it were operating
at 20 m above the ground?
(A) 15 kW
(B) 21 kW
(C) 30 kW
(D) 60 kW
(E) 120 kW

### Structured solution

Wind speed v ∝ height h (near ground). Available kinetic power flux ∝ ρ A v^3; with the same turbine (fixed A) and constant conversion fraction, electrical power P ∝ v^3 ∝ h^3. Doubling height from 10 m to 20 m doubles v, so P scales by 2^3 = 8: P = 8 × 15 kW = 120 kW. Choice (E).

### Indispensable steps

1. Use v ∝ h and P ∝ v^3
2. Scale: P ∝ h^3 ⇒ doubling h multiplies P by 8

### Independent check

Let v∝h and turbine power P∝v^3 (kinetic energy flux). Thus P∝h^3. So P(20)=15 kW*(20/10)^3=15*8=120 kW.

## 2025_fnet_ma_exam_Q20

- Answer: **C**

- Difficulty: **D3**

- Verification: `blind_answer_agreement`

### Source question

The International Space Station orbits the Earth in a circular orbit 400 km above the surface, and a full
revolution takes 93 minutes. An astronaut on a space walk neglects safety precautions and tosses away a
spanner at a speed of 1 m/s directly towards the Earth. You may assume that the Earth is a sphere of
uniform density. At which of the following five times will the spanner be closest to the astronaut?
(A) After 139.5 minutes.
(B) After 131.5 minutes.
(C) After 93 minutes.
(D) After 46.5 minutes.
(E) After 1 minute.

### Structured solution

Use the Hill/Clohessy–Wiltshire (CW) equations in the LVLH (orbiting) frame with x radial-out, y along-track. For a circular reference orbit with mean motion n = 2π/T (T = 93 min), the linear relative dynamics are
x¨ − 2n y˙ − 3n^2 x = 0,  y¨ + 2n x˙ = 0.
Initial conditions (astronaut at origin): x(0)=0, y(0)=0; a 1 m/s toss toward Earth gives x˙(0)=−1 m/s, y˙(0)=0. Solutions:
x(t) = −(1/n) sin(nt),  y(t) = (2/n)[1 − cos(nt)].
Thus relative separation ρ^2 = x^2 + y^2 = (1/n^2)[sin^2(nt) + 4(1 − cos nt)^2], which is minimized when cos nt = 1 (since 3cos^2 nt − 8 cos nt + 5 decreases with cos nt on [−1,1]). The first such time after t=0 is nt = 2π, i.e., t = 2π/n = T = 93 min. Therefore the spanner is closest (indeed co-located to first order in 1 m/s ≪ v_orb) after one orbit. Choice (C).

### Indispensable steps

1. Adopt LVLH frame and CW equations with n=2π/T
2. Apply initial radial velocity: x˙(0)=−1, others zero
3. Solve: x=−(1/n)sin nt, y=(2/n)(1−cos nt)
4. Minimize ρ^2 ⇒ first minimum at t=T=93 min

### Independent check

Linearize motion about the circular orbit (Hill's equations) with orbital angular speed Ω=2π/(93 min). With x radial (outward), y along-track, initial x(0)=y(0)=0, x'(0)=-u (u=1 m/s toward Earth), y'(0)=0. Solution: x(t)=-(u/Ω) sin(Ωt), y(t)=(2u/Ω)(1−cos(Ωt)). Distance^2∝sin^2(Ωt)+4(1−cos(Ωt))^2 which has extrema only at Ωt=nπ; t=0 is the initial minimum and the next return to minimum is at Ωt=2π, i.e. one orbital period t=93 min. Hence choice C.

## 2026_fnet_ma_exam_Q02

- Answer: **D**

- Difficulty: **D3**

- Verification: `blind_answer_agreement`

### Source question

A series of dominoes are stood upright in a line. The dominoes have mass density ρ, height h, and spacing
d between them. When the first domino is knocked over, a wave of falling dominoes propagates down the
line with speed v.

Now, a second set of dominoes is set up identically, but with all dimensions (h and d) scaled by λ. The
mass density is the same. Let v′ denote the wave speed of the scaled system. Which of the following best
describes how v′ depends on λ?
(A) v' scales as 1/sqrt(lambda)
(B) v' scales as lambda
(C) v' scales as 1/lambda
(D) v' scales as sqrt(lambda)
(E) v' is independent of lambda

### Structured solution

Let spacing-to-height ratio r = d/h be fixed under geometric scaling h→λh, d→λd. One domino step advances by distance ∼d and takes a fall time τ to reach the contact angle; by similarity, τ = C(r)√(h/g), where C(r) is dimensionless and unchanged when scaling. Hence wave speed v ∼ d/τ = [r h]/[C(r)√(h/g)] = [r/C(r)]√(g h). Under scaling, v′/v = √(h′/h) = √λ. Therefore v′ scales as √λ, i.e., choice (D).

### Indispensable steps

1. Identify that r=d/h is invariant and g is unchanged
2. Use similarity: τ ∝ √(h/g) with a constant depending only on r
3. Compute v ∝ d/τ ∝ √(g h) and infer v′/v = √λ

### Independent check

Scale: h' = λh, d' = λd, same density ⇒ mass m' ∝ λ^3. For a single domino pivoting about its base: torque τ ∼ m g h ∝ λ^3·g·λh = λ^4τ0. Moment of inertia I ∼ m h^2 ∝ λ^3·λ^2 = λ^5 I0. Angular acceleration α ∼ τ/I ∝ λ^4/λ^5 = λ^{-1}. Fall time t ∼ 1/√α ∝ √λ. Wave speed v ∼ spacing/time ∝ (λ d0)/(√λ t0) = √λ · v0. Thus v' ∝ √λ (D).

## 2026_fnet_ma_exam_Q04

- Answer: **A**

- Difficulty: **D1**

- Verification: `blind_answer_agreement`

### Source question

Two small balls are launched simultaneously from the same point at some height above horizontal ground.
One ball is launched vertically upward with speed 3 m/s, while the other is launched horizontally with
speed 4 m/s and lands on the ground at a horizontal distance of 20 m from the launch point. Neglect air
resistance. At the moment the second ball lands, how far is it from the first ball?
(A) 25 m
(B) 36 m
(C) 60 m
(D) 80 m
(E) 240 m

### Structured solution

Let t be when the horizontal-launch ball lands. Horizontal motion gives t = 20/4 = 5 s. With upward positive and same initial height y0 for both, the landing condition for the horizontal-launch ball is y0 − (1/2) g t^2 = 0 ⇒ y0 = (1/2) g t^2. The vertical-launch ball’s height then is y1(t) = y0 + (3)t − (1/2) g t^2 = (1/2) g t^2 + 3t − (1/2) g t^2 = 3t = 15 m. Horizontal separation is 20 m, vertical separation is 15 m, so distance = √(20^2 + 15^2) = 25 m ⇒ (A).

### Indispensable steps

1. Use x = vxt to find t = 5 s for the horizontal-launch ball
2. Eliminate y0 using the horizontal-launch landing condition
3. Compute the vertical position of the vertical-launch ball at t and take Euclidean distance

### Independent check

Take +y up. Second (horizontal) ball: vx=4 m/s, lands at x=20 ⇒ t = 20/4 = 5 s. Its vertical motion gives initial height y0 from 0 = y0 - ½ g t^2 ⇒ y0 = ½ g t^2 = 0.5·9.8·25 = 122.5 m. First ball (vertical, v0=+3 m/s): at t=5, y1 = y0 + v0 t - ½ g t^2 = y0 +3·5 -½ g·25 = y0 +15 - y0 = 15 m. Horizontal separation =20 m, vertical separation =15 m → distance = √(20^2+15^2)=25 m (A).

## 2026_fnet_ma_exam_Q12

- Answer: **B**

- Difficulty: **D2**

- Verification: `blind_answer_agreement`

### Source question

A wake surfer of total mass M (surfer plus board) is being towed at a constant horizontal velocity v across
a flat lake. The wakeboard has a specific geometry such that, at this speed, it is partially submerged and
provides a static buoyant force FB (where FB < Mg).

To support the remainder of the weight, the board moves with an angle of attack θ relative to the horizontal
water surface. Assume that the water exerts a reaction force strictly normal to the bottom surface of the
board. Which of the following expressions represents the horizontal tension T in the tow rope required to
maintain this constant velocity?
(A) T = Mg tan θ
(B) T = (Mg −FB) tan θ
(C) T = (Mg −FB) sin θ
(D) T = Mg sin θ
(E) T = FB cos θ

### Structured solution

Choose +x horizontal (tow direction), +y upward. Forces: T in +x; weight Mg in −y; buoyant force FB in +y; water reaction N normal to the board. If the board is pitched at angle θ above horizontal, the normal has components N_x = −N sinθ, N_y = N cosθ. Steady tow (no acceleration) gives ΣFx = 0 ⇒ T + (−N sinθ) = 0 ⇒ T = N sinθ, and ΣFy = 0 ⇒ N cosθ + FB − Mg = 0 ⇒ N = (Mg − FB)/cosθ. Thus T = (Mg − FB) tanθ, choice (B).

### Indispensable steps

1. Resolve the water reaction N into components: N_x = −N sinθ, N_y = N cosθ
2. Apply ΣFx = 0 and ΣFy = 0 to solve for N and then T

### Independent check

Forces: weight Mg down, buoyant FB up, hydrodynamic normal R acting perpendicular to board (board at angle θ to horizontal). Resolve components: vertical equilibrium: R cosθ + FB = Mg ⇒ R cosθ = Mg − FB. Horizontal equilibrium (forward positive): T − R sinθ = 0 ⇒ T = R sinθ. Eliminate R: T = (Mg − FB) tanθ. Choice (B).

## 2024_fnet_ma_exam_Q11

- Answer: **A**

- Difficulty: **D2**

- Verification: `manual_equation_review`

### Source question

Two hemispherical shells can be pressed together to form a airtight sphere of radius 40 cm. Suppose the
shells are pressed together at a high altitude, where the air pressure is half its value at sea level. The
sphere is then returned to sea level, where the air pressure is 10^5 Pa. What force F, applied directly
outward to each hemisphere, is required to pull them apart?
(A) 25,000 N
(B) 50,000 N
(C) 100,000 N
(D) 200,000 N
(E) 400,000 N

### Structured solution

Inside pressure is sealed at high altitude: pin = (1/2) p0 = 5×10^4 Pa. At sea level pout = p0 = 10^5 Pa. Pressure difference Δp = pout − pin = 5×10^4 Pa compresses the sphere. The net force holding each hemisphere on is the resultant over the hemisphere, equal to Δp times the great-circle area A = π R^2 with R=0.40 m: F = Δp π R^2 = (5×10^4)×π×(0.40)^2 ≈ 2.51×10^4 N ≈ 25,000 N. This is the outward force per hemisphere required to separate. Choice mapping: (A).

### Indispensable steps

1. Note pin = 0.5 p0; compute Δp = p0 − pin
2. Use resultant pressure force on a hemisphere F = Δp·πR^2
3. Evaluate with R=0.40 m

### Independent check

The sealed interior remains at 5e4 Pa while sea-level exterior pressure is 1e5 Pa. The compressive resultant on either hemisphere is pressure difference times projected great-circle area: F=(5e4) pi (0.40)^2=2.51e4 N, choice A.

## 2024_fnet_ma_exam_Q21

- Answer: **D**

- Difficulty: **D3**

- Verification: `manual_equation_review`

### Source question

A syringe is filled with water of density ρ and negligible viscosity. Its body is a cylinder of cross-sectional
area A1, which gradually tapers into a needle with cross-sectional area A2 ≪A1. The syringe is held in
place and its end is slowly pushed inward by a force F, so that it moves with constant speed v. Water
shoots straight out of the needle’s tip. What is the approximate value of F?
(A) rho v^2 A1
(B) rho v^2 A1^2/(2 A2)
(C) rho v^2 A1^2/A2
(D) rho v^2 A1^3/(2 A2^2)
(E) rho v^2 A1^3/A2^2

### Structured solution

Steady, incompressible, negligible viscosity; syringe held fixed. Piston speed v gives barrel speed v1=v and needle speed v2=u via continuity: u = v A1/A2. Apply Bernoulli from barrel to just outside tip (height change negligible): p1 + (1/2)ρ v^2 = patm + (1/2)ρ u^2 ⇒ p1 − patm = (1/2)ρ(u^2 − v^2). Force on piston equals gauge pressure times area: F = (p1 − patm) A1 ≈ (1/2)ρ A1 (v^2 (A1^2/A2^2)) = (1/2) ρ v^2 A1^3 / A2^2, since A2 ≪ A1 ⇒ u^2 ≫ v^2. Choice mapping: (D).

### Indispensable steps

1. Use continuity: u = v A1/A2
2. Apply Bernoulli: p1 − patm = (1/2)ρ(u^2 − v^2)
3. Relate force: F = (p1 − patm) A1
4. Approximate u^2 ≫ v^2 for A2 ≪ A1 to get F ≈ (1/2) ρ v^2 A1^3/A2^2

### Independent check

Continuity gives needle speed u=v A1/A2. Bernoulli gives gauge pressure approximately rho u^2/2, so piston force is F approximately (rho/2)(v A1/A2)^2 A1=rho v^2 A1^3/(2 A2^2), choice D.

## 2026_fnet_ma_exam_Q15

- Answer: **B**

- Difficulty: **D2**

- Verification: `manual_equation_review`

### Source question

A ball launcher fires balls along the floor at some initial speed, applying no rotation to them. The balls
initially slip along the floor, then start rolling without slipping. Ignore the potential deformation of the
ball and flooring during this process, as well as air resistance. How does the final speed of the rolling ball
depend on the coefficient of friction µ between the ball and the floor?
(A) The final speed is larger when µ is large
(B) The final speed is the same regardless of µ
(C) The final speed is larger when µ is small
(D) The final speed is larger when µ is small for high launch speeds, and when µ is large for low launch speeds
(E) The final speed is larger when µ is large for high launch speeds, and when µ is small for low launch speeds

### Structured solution

Frame and signs: inertial ground frame; +x along launch direction. Take angular velocity ω positive so that friction torque that spins the ball up is +. Data: at t=0, v(0)=v0>0, ω(0)=0. While slipping, kinetic friction of magnitude f=μmg acts backward (−x), giving m dv/dt=−μmg and I dω/dt=+fR=+μmgR. Rolling threshold: slip stops when v=ωR (since initially v−ωR>0, friction opposes the forward slip). Time to reach rolling: v(t)=v0−μgt, ω(t)=(μmgR/I)t ⇒ enforce v0−μgt*=R(μmgR/I)t* ⇒ v0=μg t* [1+mR^2/I] ⇒ t*=v0/[μg(1+mR^2/I)]. Final rolling speed: v_f=v(t*)=v0−μg t*=v0[1−1/(1+mR^2/I)]=v0[mR^2/(I+mR^2)], which is independent of μ. After rolling begins on a horizontal floor, static friction can be zero; v and ω remain constant. Choice mapping: independent of μ ⇒ (B).

### Indispensable steps

1. Write m dv/dt=−μmg and I dω/dt=+μmgR while slipping
2. Impose rolling condition v=ωR to find t*
3. Evaluate v_f=v0−μg t* and note μ cancels

### Independent check

During slipping, dv/dt=-mu g and d omega/dt=mu m g R/I. Applying v=omega R at the rolling transition makes the transition time proportional to 1/mu, so mu cancels from v_final. Choice B.

## 2026_fnet_ma_exam_Q17

- Answer: **D**

- Difficulty: **D4**

- Verification: `manual_equation_review`

### Source question

A ladder is leaning against a vertical wall. The ladder is a uniform rod of mass M and length L, and both
the wall and the ground are frictionless. The ladder is released from rest from an almost-vertical position
and begins to slide. What is the speed of the point of the ladder that is in contact with the floor when it
is a horizontal distance sqrt(3)L/2 away from the wall?
(A) 0.46 sqrt(gL)
(B) 0.51 sqrt(gL)
(C) 0.56 sqrt(gL)
(D) 0.61 sqrt(gL)
(E) 0.66 sqrt(gL)

### Structured solution

Frame: origin at floor-wall corner; x along floor, y upward. Let θ be the angle between the ladder and the floor, so x=L cosθ, y=L sinθ. Contacts are frictionless: the floor normal is vertical while the foot’s velocity is horizontal; the wall normal is horizontal while the top’s velocity is vertical. Hence contact forces do no work; mechanical energy is conserved. Kinematics: ẋ=−L sinθ θ̇, ẏ=L cosθ θ̇. CM height y_cm=y/2=(L/2)sinθ, so U=Mg(L/2)sinθ. The CM speed satisfies V_cm^2=(ẋ^2+ẏ^2)/4=L^2 θ̇^2/4. For a uniform rod I_cm=(1/12)ML^2, so T=(1/2)M V_cm^2+(1/2)I_cm θ̇^2=ML^2 θ̇^2/6. Released from rest near θ≈π/2, energy gives Mg(L/2)(1−sinθ)=ML^2 θ̇^2/6 ⇒ θ̇^2=(3g/L)(1−sinθ). The foot speed is v_floor=|ẋ|=L sinθ |θ̇|, so v_floor^2=3gL sin^2θ(1−sinθ). At x=√3 L/2, cosθ=√3/2 ⇒ sinθ=1/2, hence v_floor^2=3gL(1/4)(1/2)=3gL/8 and v_floor=√(3/8)√(gL)≈0.61√(gL). Choice (D).

### Indispensable steps

1. Note contact forces do no work ⇒ use energy conservation
2. Express T in terms of θ̇: T=ML^2 θ̇^2/6; U=Mg(L/2)sinθ
3. Solve θ̇^2=(3g/L)(1−sinθ) and use v_floor=|ẋ|=L sinθ |θ̇| at sinθ=1/2

### Independent check

With x=L cos(theta), energy gives theta_dot^2=(3g/L)(1-sin(theta)). At x=sqrt(3)L/2, sin(theta)=1/2. The foot speed L sin(theta)|theta_dot| is sqrt(3gL/8)=0.612 sqrt(gL), choice D.

## 2026_fnet_ma_exam_Q21

- Answer: **C**

- Difficulty: **D4**

- Verification: `manual_equation_review`

### Source question

A student stands on a large horizontal merry-go-round (R = 2.0 m) at an initial radius of r0 = 1.0 m.
Both rotate at constant angular speed ω = 1.2 rad/s. The student wants to get off without walking and
performs a sequence of identical vertical jumps of height h = 0.31 m. Ignore air resistance.

What is the minimum number of jumps needed for the student to land off the platform? You may assume
that the merry-go-round is much more massive than the student and that friction instantly brings the
student back into co-rotation with the platform.
(A) 3
(B) 4
(C) 5
(D) 6
(E) It is impossible to move outward by purely vertical jumps

### Structured solution

Frame: inertial, origin at carousel center, +θ CCW. At takeoff from radius r, the student’s horizontal velocity equals the platform’s tangential speed v=ωr (purely +θ). While airborne (no horizontal forces), horizontal position evolves as x=r (radial line at takeoff), y=ωrt, so the radial distance grows as r(t)=√(r^2+(ωrt)^2)=r√(1+(ωt)^2). Flight time for a vertical jump of height h is T=2√(2h/g). Upon landing, friction instantaneously reattaches the student to co-rotate at the same ω at the new radius. Hence each jump multiplies radius by f=√(1+(ωT)^2). After n identical jumps: rn=r0 f^n. The student lands off when rn≥R, i.e., n≥ln(R/r0)/ln f. Numbers: h=0.31 m ⇒ T=2√(2h/g)=2√(0.62/9.8)=0.503 s; ωT=1.2×0.503=0.604; f=√(1+0.604^2)=1.168. Then n≥ln(2/1)/ln(1.168)=0.6931/0.1553=4.46 ⇒ minimum integer n=5. Check: after 4 jumps r≈1.86<2; after 5 r≈2.17>2. Choice (C).

### Indispensable steps

1. Model airborne motion in inertial frame: horizontal speed constant v=ωr, giving r′=r√(1+(ωT)^2) per jump
2. Compute T from h: T=2√(2h/g)
3. Apply multiplicative growth rn=r0[√(1+(ωT)^2)]^n and solve rn≥R for n

### Independent check

Each jump lasts T=2 sqrt(2h/g). In flight the inertial tangential velocity is fixed, so the landing radius is multiplied by sqrt(1+(omega T)^2)=1.168. The least n with 1.0(1.168)^n at least 2.0 is n=5, choice C.

## 2026_fnet_ma_exam_Q22

- Answer: **A**

- Difficulty: **D4**

- Verification: `manual_equation_review`

### Source question

A tricycle of mass m = 100 kg is traveling north on a horizontal surface. The geometry of the tricycle is
defined as follows:

• Wheelbase (distance from front axle to rear axle): L = 1.0 m

• Rear track width (distance between rear wheels): d = 1.0 m

• Center of mass (CM) location: on the longitudinal symmetry axis, a distance b = 0.4 m forward of
the rear axle

• Radius of gyration about the CM: k = 0.5 m

(The radius of gyration k is defined by ICM = mk2, where ICM is the moment of inertia of the tricycle
about a vertical axis through its center of mass.)

Seeing a patch of ice on their right side ahead, the rider panics and slams on the brakes. The front wheel
and the left rear wheel slide on dry pavement with a coefficient of kinetic friction µk = 0.5, while the right
rear wheel slides on frictionless ice (µ = 0).

This asymmetry in friction forces produces a net torque about the center of mass, causing the tricycle to
begin rotating. Calculate the initial angular acceleration α of the tricycle.
(A) 3.0 rad/s2, Counter-Clockwise
(B) 3.0 rad/s2, Clockwise
(C) 3.3 rad/s2, Counter-Clockwise
(D) 6.0 rad/s2, Counter-Clockwise
(E) 3.3 rad/s2 Clockwise

### Structured solution

Choose x east, y north, z up; CCW about +z is positive. Geometry relative to CM: rear axle at y=−b=−0.4 m; front axle at y=L−b=0.6 m; rear track half-width d/2=0.5 m, so left rear at (x, y)=(-0.5, -0.4), right rear at (+0.5, -0.4), front at (0, 0.6). Vertical loads (no height given ⇒ use static distribution): N_F=mg·b/L=0.4mg, N_R=mg−N_F=0.6mg split equally ⇒ N_LR=N_RR=0.3mg. Sliding wheels: front and left rear with μk=0.5; right rear on ice (μ=0). Friction forces oppose motion (south): F_F=μk N_F=0.5·0.4mg=0.2mg, F_LR=μk N_LR=0.5·0.3mg=0.15mg; both act along −y. Torque about CM: τz=Σ(x Fy − y Fx)=Σ x Fy since Fx=0. Front has x=0 ⇒ no yaw torque. Left rear: x=−d/2, Fy=−F_LR ⇒ τz= (−0.5)(−0.15mg)=+0.075mg. Numerically, mg≈980 N ⇒ τz=0.075·980=73.5 N·m (CCW). Moment of inertia about CM: I=mk^2=100·(0.5)^2=25 kg·m^2. Thus α=τz/I=73.5/25=2.94 rad/s^2 ≈ 3.0 rad/s^2 CCW. Choice (A).

### Indispensable steps

1. Compute normal loads: N_F=mg b/L; N_R=mg−N_F, split rear equally
2. Apply kinetic friction on sliding wheels: F_F=μk N_F, F_LR=μk(N_R/2), directions −y
3. Sum yaw moments about CM: τz=Σ x Fy; only left rear contributes
4. Use I=mk^2 to get α=τz/I and assign CCW sign

### Independent check

Static loads are 0.4mg front and 0.3mg on each rear wheel. Only left-rear friction produces yaw torque: tau=(d/2) mu (0.3mg)=73.5 N m counterclockwise. With I=mk^2=25 kg m^2, alpha=2.94 rad/s^2 counterclockwise, choice A.
