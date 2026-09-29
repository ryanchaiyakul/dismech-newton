# Alternating Direction Method of Multipliers (ADMM)

Consider the split optimization problems of the form:

$$
\min_{\mathbf{x}, \mathbf{z}} \; f(\mathbf{x}) + g(\mathbf{z}) \quad \text{s.t.} \quad \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} = \mathbf{c}
$$

The Augmented Lagrangian for this system is:

$$
\mathcal{L}_{\rho}(\mathbf{x}, \mathbf{z}, \mathbf{u}) = f(\mathbf{x}) + g(\mathbf{z}) + \frac{\rho}{2} \Vert{}\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c} + \mathbf{u}\Vert{}_2^2 - \frac{\rho}{2} \Vert{}\mathbf{u}\Vert{}_2^2
$$

where $\mathbf{u}$ is the scaled dual variable and $\rho > 0$ is the penalty parameter.

### Update Steps

To solve the Augmented Lagrangian $\mathcal{L}_{\rho}(\mathbf{x}, \mathbf{z}, \mathbf{u})$, ADMM alternates between minimizing over primal variables $(\mathbf{x}, \mathbf{z})$ and performing dual ascent on $\mathbf{u}$:

1. **Primal Step 1 ($\mathbf{x}$-update):**
$$\mathbf{x}^{(k+1)} = \arg\min_{\mathbf{x}} \; f(\mathbf{x}) + \frac{\rho}{2} \left\Vert{} \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z}^{(k)} - \mathbf{c} + \mathbf{u}^{(k)} \right\Vert{}_2^2$$

2. **Primal Step 2 ($\mathbf{z}$-update):**
$$\mathbf{z}^{(k+1)} = \arg\min_{\mathbf{z}} \; g(\mathbf{z}) + \frac{\rho}{2} \left\Vert{} \mathbf{A}\mathbf{x}^{(k+1)} + \mathbf{B}\mathbf{z} - \mathbf{c} + \mathbf{u}^{(k)} \right\Vert{}_2^2$$

3. **Dual Update ($\mathbf{u}$-update):**
$$\mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} + \left( \mathbf{A}\mathbf{x}^{(k+1)} + \mathbf{B}\mathbf{z}^{(k+1)} - \mathbf{c} \right)$$

where $k$ denotes the ADMM iteration index within the current time step.


# Case Study: Quasi-static Spring-Mass System

Based on Hamilton's principle, at every time step $\Delta t$, the system reaches an equilibrium that minimizes the total energy functional:

$$
\argmin_x T(x) + V(x)
$$

where:
- $T(\mathbf{x}) = \frac{1}{2\Delta t^2} \Vert{}\mathbf{x} - \mathbf{y}\Vert{}_{\mathbf{M}}^2$ with inertial prediction $\mathbf{y} = \mathbf{x}^{(t)} + \Delta t \mathbf{v}^{(t)} + \Delta t^2 \mathbf{g}$.
- $V(\mathbf{x}) = \sum_{m} W_m(\mathbf{x}_j - \mathbf{x}_i)$ is the total potential energy.

## Variable Splitting

$V(\mathbf{x})$ consists of independent local energy evaluations over stencil displacement vectors $(\mathbf{x}_j - \mathbf{x}_i)$. To exploit this separability, we introduce local auxiliary deformation variables $\mathbf{z}_m$:

$$V(\mathbf{z}) = \sum_{m} W_m(\mathbf{z}_m)$$

The deformation variable $\mathbf{z}_m$ generalizes beyond two-node edges to encompass arbitrary local stencils.

Matching terms to standard ADMM form:

$$f(\mathbf{x}) = \frac{1}{2\Delta t^2} \Vert{}\mathbf{x} - \mathbf{y}\Vert{}_{\mathbf{M}}^2, \quad g(\mathbf{z}) = \sum_{m} W_m(\mathbf{z}_m)$$

Setting $\mathbf{A} = \mathbf{S}$ (spatial incidence matrix), $\mathbf{B} = -\mathbf{I}$, and $\mathbf{c} = \mathbf{0}$ yields the linear coupling constraint:

$$\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} = \mathbf{c} \quad \implies \quad \mathbf{S}\mathbf{x} - \mathbf{z} = \mathbf{0}$$

This yields the split optimization problem:

$$\min_{\mathbf{x}, \mathbf{z}} \; \frac{1}{2\Delta t^2} \Vert{}\mathbf{x} - \mathbf{y}\Vert{}_{\mathbf{M}}^2 + \sum_{m} W_m(\mathbf{z}_m) \quad \text{s.t.} \quad \mathbf{S}\mathbf{x} - \mathbf{z} = \mathbf{0}$$

The corresponding Augmented Lagrangian is:

$$\mathcal{L}_{\rho}(\mathbf{x}, \mathbf{z}, \mathbf{u}) = \frac{1}{2\Delta t^2} \Vert{}\mathbf{x} - \mathbf{y}\Vert{}_{\mathbf{M}}^2 + \sum_{m} W_m(\mathbf{z}_m) + \frac{\rho}{2} \Vert{}\mathbf{S}\mathbf{x} - \mathbf{z} + \mathbf{u}\Vert{}_2^2 - \frac{\rho}{2} \Vert{}\mathbf{u}\Vert{}_2^2$$

### Update Steps

1. **Global Step ($\mathbf{x}$-update):**

$$\mathbf{x}^{(k+1)} = \arg\min_{\mathbf{x}} \; \frac{1}{2\Delta t^2} \Vert{}\mathbf{x} - \mathbf{y}\Vert{}_{\mathbf{M}}^2 + \frac{\rho}{2} \left\Vert{} \mathbf{S}\mathbf{x} - \mathbf{z}^{(k)} + \mathbf{u}^{(k)} \right\Vert{}_2^2$$

2. **Local Step ($\mathbf{z}$-update):**
$$\mathbf{z}^{(k+1)} = \arg\min_{\mathbf{z}} \; \sum_{m} W_m(\mathbf{z}_m) + \frac{\rho}{2} \left\Vert{} \mathbf{S}\mathbf{x}^{(k+1)} - \mathbf{z} + \mathbf{u}^{(k)} \right\Vert{}_2^2$$

3. **Dual Update ($\mathbf{u}$-update):**
$$\mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} + \left( \mathbf{S}\mathbf{x}^{(k+1)} - \mathbf{z}^{(k+1)} \right)$$

## Subproblem Optimization

### Local Step ($\mathbf{z}$-update)

Because both the potential $W_m$ and the penalty term decouple across individual elements, the minimization evaluates independently for each element $m$:

$$\mathbf{z}_m^{(k+1)} = \arg\min_{\mathbf{z}_m} \; W_m(\mathbf{z}_m) + \frac{\rho}{2} \left\Vert{} \mathbf{S}_m \mathbf{x}^{(k+1)} - \mathbf{z}_m + \mathbf{u}_m^{(k)} \right\Vert{}_2^2$$

Because $\mathbf{z}_m$ operates locally per stencil, each element subproblem is independent. Even with overlapping stencils, each local solve remains small, fixed in dimension, and embarrassingly parallel.

#### Closed-Form Damped Hookean Linear Spring

Consider a Hookean spring with stiffness $k_m$, rest length $r_m$, and damping coefficient $c_m$ acting on the rate of change of its length $\ell_m = \Vert{}\mathbf{z}_m\Vert{}_2$, i.e. the axial force $-\left(k_m (\ell_m - r_m) + c_m \dot{\ell}_m\right)$. The damping force derives from the Rayleigh dissipation potential $\mathcal{D}_m = \frac{1}{2} c_m \dot{\ell}_m^2$. Discretizing with backward Euler,

$$\dot{\ell}_m \approx \frac{\Vert{}\mathbf{z}_m\Vert{}_2 - \ell_m^{(t)}}{\Delta t}, \qquad \ell_m^{(t)} = \Vert{}\mathbf{S}_m \mathbf{x}^{(t)}\Vert{}_2,$$

the incremental dissipation $\Delta t \, \mathcal{D}_m$ is added to the elastic energy, giving the per-step local potential:

$$W_m(\mathbf{z}_m) = \frac{1}{2} k_m \left(\Vert{}\mathbf{z}_m\Vert{}_2 - r_m\right)^2 + \frac{c_m}{2\Delta t} \left(\Vert{}\mathbf{z}_m\Vert{}_2 - \ell_m^{(t)}\right)^2$$

Since $\ell_m^{(t)}$ is fixed over the time step, the local step remains separable. Both terms depend only on $\Vert{}\mathbf{z}_m\Vert{}_2$, so the optimal direction is that of the target displacement $\mathbf{d}_m = \mathbf{S}_m \mathbf{x}^{(k+1)} + \mathbf{u}_m^{(k)}$. Writing $\mathbf{z}_m = \ell \, \hat{\mathbf{d}}_m$ and setting the derivative with respect to $\ell$ to zero:

$$k_m (\ell - r_m) + \frac{c_m}{\Delta t} (\ell - \ell_m^{(t)}) + \rho (\ell - \Vert{}\mathbf{d}_m\Vert{}_2) = 0$$

yields a scalar length projection:

$$\mathbf{z}_m^{(k+1)} = \left( \frac{\rho \Vert{}\mathbf{d}_m\Vert{}_2 + k_m r_m + \frac{c_m}{\Delta t} \ell_m^{(t)}}{\rho + k_m + \frac{c_m}{\Delta t}} \right) \frac{\mathbf{d}_m}{\Vert{}\mathbf{d}_m\Vert{}_2}$$

In effect, the damped spring acts as an undamped spring with stiffness $k_m + c_m / \Delta t$ whose rest length is pulled toward the start-of-step length $\ell_m^{(t)}$. Setting $c_m = 0$ recovers the undamped Hookean projection $\left(\rho \Vert{}\mathbf{d}_m\Vert{}_2 + k_m r_m\right) / \left(\rho + k_m\right)$. Damping only enters the local step, so the global system matrix $\mathbf{H}$ is unchanged.

### Global Step ($\mathbf{x}$-update)

Differentiating the quadratic global objective with respect to $\mathbf{x}$ and setting the gradient to zero:

$$\nabla_{\mathbf{x}} \left( \frac{1}{2\Delta t^2} (\mathbf{x} - \mathbf{y})^T \mathbf{M} (\mathbf{x} - \mathbf{y}) + \frac{\rho}{2} (\mathbf{S}\mathbf{x} - \mathbf{z}^{(k)} + \mathbf{u}^{(k)})^T (\mathbf{S}\mathbf{x} - \mathbf{z}^{(k)} + \mathbf{u}^{(k)}) \right) = \mathbf{0}$$
Yields the sparse linear system:

$$\underbrace{\left( \frac{\mathbf{M}}{\Delta t^2} + \rho \mathbf{S}^T \mathbf{S} \right)}_{\mathbf{H}} \mathbf{x}^{(k+1)} = \frac{\mathbf{M}}{\Delta t^2} \mathbf{y} + \rho \mathbf{S}^T \left( \mathbf{z}^{(k)} - \mathbf{u}^{(k)} \right)$$

The system matrix $\mathbf{H} = \frac{\mathbf{M}}{\Delta t^2} + \rho \mathbf{S}^T \mathbf{S}$ depends solely on mass $\mathbf{M}$, time step $\Delta t$, penalty parameter $\rho$, and topology matrix $\mathbf{S}$. If these parameters remain constant, $\mathbf{H}$ can be pre-factorized once at startup. During simulation, solving $\mathbf{H}\mathbf{x^{(k+1)}}=\mathbf{b}$ reduces to a back-substitution step at $\mathcal{O}(nnz(\mathbf{L}))$ where $\mathbf{L}$ is the number of non-zero entries in the triangular factor matrix.

## Contact: Multiple Local Steps

To add contact and friction into our variational system, we elevate the unconstrained energy minimization problem to a constrained optimization problem defined over the non-convex set of admissible states.

$$\arg\min_{\mathbf{x} \in \mathbb{R}^{3N}} \quad T(\mathbf{x}) + V(\mathbf{x}) \quad \text{s.t.} \quad \mathbf{C}_c \mathbf{x} \in \mathcal{C}_c, \quad \forall c \in \{1, \dots, M_c\}$$

Let $M_c$ define the set of active contact stencils (i.e. Vertex-Face, Edge-Edge, Point-Edge) identified by a collision detection pipeline. For each active contact $c\in \{1,\dots,M_c\}$, let:
- $\mathbf{C}_c\in\mathbb{R}^{3\times3N}$ be the linear matrix mapping DOFs to the local contact frame.
- $\mathcal{C}_c\subset\mathbb{R}^3$ be the local Signorini-Coulomb feasible set.

We can convert this hard-constrained optimization into an unconstrained problem over $\mathbb{R}^{3N}$ by introducing the indicator function $\delta_{\mathcal{C}_c}(\mathbf{C}_c\mathbf{x})$ over the admissible contact set $\mathcal{C}_c$:

$$\delta_{\mathcal{C}_c}(\mathbf{z}_c) =  \begin{cases}  0 & \text{if } \mathbf{z}_c \in \mathcal{C}_c \\ +\infty & \text{if } \mathbf{z}_c \notin \mathcal{C}_c \end{cases}$$

The unconstrained variational formulation becomes: 

$$\arg\min_{\mathbf{x}} \; T(\mathbf{x}) + V(\mathbf{x}) + \sum_{c=1}^{M_c} \delta_{\mathcal{C}_c}(\mathbf{C}_c \mathbf{x})$$

Where:
- $T(\mathbf{x}) = \frac{1}{2\Delta t^2} \Vert{}\mathbf{x} - \mathbf{y}\Vert{}_{\mathbf{M}}^2$ with inertial prediction $\mathbf{y} = \mathbf{x}^{(t)} + \Delta t \mathbf{v}^{(t)} + \Delta t^2 \mathbf{g}$.
- $V(\mathbf{x}) = \sum_{m=1}^{M_s} W_m(\mathbf{S}_m \mathbf{x})$ is the total potential energy.
- $M_s$.$\delta_{\mathcal{C}_c}(\cdot)$  is the infinite barrier penalty.

### Variable Splitting

Evaluating the indicator function $\delta_{\mathcal{C}_c}(\mathbf{C}_c \mathbf{x})$ directly on the global degrees of freedom $\mathbf{x}$ tightly couples all overlapping contact stencils across the entire mesh. To decouple this global constraint, we introduce a local auxiliary contact deformation variable $\mathbf{z}_{c,c} \in \mathbb{R}^3$ for each active contact $c \in \{1, \dots, M_c\}$.

We group all local contact target vectors into a single concatenated vector $\mathbf{z}_c \in \mathbb{R}^{3 M_c}$:

$$\mathbf{z}_c = \begin{bmatrix} \mathbf{z}_{c, 1} \\ \mathbf{z}_{c, 2} \\ \vdots \\ \mathbf{z}_{c, M_c} \end{bmatrix}, \quad \text{and} \quad \mathbf{C} = \begin{bmatrix} \mathbf{C}_1 \\ \mathbf{C}_2 \\ \vdots \\ \mathbf{C}_{M_c} \end{bmatrix} \in \mathbb{R}^{3 M_c \times 3N}$$

This allows us to split the non-smooth contact energy potential into an independent, element-wise sum over local target variables:

$$g_{\text{contact}}(\mathbf{z}_c) = \sum_{c=1}^{M_c} \delta_{\mathcal{C}_c}(\mathbf{z}_{c,c})$$

Combining this contact target vector $\mathbf{z}_c$ with our internal elastic stencil vector $\mathbf{z}_s \in \mathbb{R}^{3 M_s}$ yields the complete split optimization problem:

$$\min_{\mathbf{x}, \mathbf{z}_s, \mathbf{z}_c} \quad \frac{1}{2\Delta t^2} \Vert{}\mathbf{x} - \mathbf{y}\Vert{}_{\mathbf{M}}^2 + \sum_{m=1}^{M_s} W_m(\mathbf{z}_{s,m}) + \sum_{c=1}^{M_c} \delta_{\mathcal{C}_c}(\mathbf{z}_{c,c}) \quad \text{s.t.} \quad \begin{bmatrix} \mathbf{S} \\ \mathbf{C} \end{bmatrix} \mathbf{x} - \begin{bmatrix} \mathbf{z}_s \\ \mathbf{z}_c \end{bmatrix} = \mathbf{0}$$

Assigning penalty parameters $\rho_s > 0$ for elasticity and $\rho_c > 0$ for contact, along with scaled dual variables $\mathbf{u}_s$ and $\mathbf{u}_c$, the Augmented Lagrangian $\mathcal{L}_{\boldsymbol{\rho}}$ is written as:

$$\mathcal{L}_{\boldsymbol{\rho}}(\mathbf{x}, \mathbf{z}_s, \mathbf{z}_c, \mathbf{u}_s, \mathbf{u}_c) = \frac{1}{2\Delta t^2} \Vert{}\mathbf{x} - \mathbf{y}\Vert{}_{\mathbf{M}}^2 + \sum_{m=1}^{M_s} W_m(\mathbf{z}_{s,m}) + \sum_{c=1}^{M_c} \delta_{\mathcal{C}_c}(\mathbf{z}_{c,c}) + \frac{\rho_s}{2}\left\Vert{} \mathbf{S}\mathbf{x} - \mathbf{z}_s + \mathbf{u}_s \right\Vert{}_2^2 + \frac{\rho_c}{2}\left\Vert{} \mathbf{C}\mathbf{x} - \mathbf{z}_c + \mathbf{u}_c \right\Vert{}_2^2 - \frac{\rho_s}{2}\Vert{}\mathbf{u}_s\Vert{}_2^2 - \frac{\rho_c}{2}\Vert{}\mathbf{u}_c\Vert{}_2^2$$

### Update Steps

1. **Global Step ($\mathbf{x}$-update):**

$$\mathbf{x}^{(k+1)} = \arg\min_{\mathbf{x}} \; \frac{1}{2\Delta t^2} \Vert{}\mathbf{x} - \mathbf{y}\Vert{}_{\mathbf{M}}^2 + \frac{\rho_s}{2} \left\Vert{} \mathbf{S}\mathbf{x} - \mathbf{z}_s^{(k)} + \mathbf{u}_s^{(k)} \right\Vert{}_2^2 + \frac{\rho_c}{2} \left\Vert{} \mathbf{C}\mathbf{x} - \mathbf{z}_c^{(k)} + \mathbf{u}_c^{(k)} \right\Vert{}_2^2$$

2. **Local Elasticity Step ($\mathbf{z}_s$-update):**

$$\mathbf{z}_s^{(k+1)} = \arg\min_{\mathbf{z}_s} \; \sum_{m=1}^{M_s} W_m(\mathbf{z}_{s,m}) + \frac{\rho_s}{2} \left\Vert{} \mathbf{S}\mathbf{x}^{(k+1)} - \mathbf{z}_s + \mathbf{u}_s^{(k)} \right\Vert{}_2^2$$

3. **Local Contact Step ($\mathbf{z}_c$-update):**

$$\mathbf{z}_c^{(k+1)} = \arg\min_{\mathbf{z}_c} \; \sum_{c=1}^{M_c} \delta_{\mathcal{C}_c}(\mathbf{z}_{c,c}) + \frac{\rho_c}{2} \left\Vert{} \mathbf{C}\mathbf{x}^{(k+1)} - \mathbf{z}_c + \mathbf{u}_c^{(k)} \right\Vert{}_2^2$$

4. **Dual Updates ($\mathbf{u}_s, \mathbf{u}_c$-updates):**

$$\mathbf{u}_s^{(k+1)} = \mathbf{u}_s^{(k)} + \left( \mathbf{S}\mathbf{x}^{(k+1)} - \mathbf{z}_s^{(k+1)} \right)$$

$$\mathbf{u}_c^{(k+1)} = \mathbf{u}_c^{(k)} + \left( \mathbf{C}\mathbf{x}^{(k+1)} - \mathbf{z}_c^{(k+1)} \right)$$

## Subproblem Optimization

### Contact Local Step ($\mathbf{z}_c$-update)

Because both the indicator penalty $\delta_{\mathcal{C}_c}$ and the quadratic penalty decouple across individual contacts, the minimization evaluates independently for each active contact $c \in \{1, \dots, M_c\}$:

$$\mathbf{z}_{c,c}^{(k+1)} = \arg\min_{\mathbf{z}_{c,c}} \; \delta_{\mathcal{C}_c}(\mathbf{z}_{c,c}) + \frac{\rho_c}{2} \left\Vert{} \mathbf{C}_c \mathbf{x}^{(k+1)} - \mathbf{z}_{c,c} + \mathbf{u}_{c,c}^{(k)} \right\Vert{}_2^2=\mathrm{Proj}_{\mathcal{C}_c}\left(\mathbf{p}_c\right)$$

where $\mathbf{p}_c = \mathbf{C}_c \mathbf{x}^{(k+1)} + \mathbf{u}_{c,c}^{(k)} \in \mathbb{R}^3$ is the local target vector.

#### Closed-Form Projection onto the de Saxcé Cone ($\mathcal{C}_c$)

Decomposing $\mathbf{p}_c$ into normal $p_n = \mathbf{p}_c \cdot \mathbf{n}_c$ and tangential $\mathbf{p}_t = \mathbf{p}_c - p_n \mathbf{n}_c$ components in the contact frame yields the Second-Order Cone (SOC) operator:

> #### Normal Step (Signorini Non-Penetration):
>
> $$z_n^{(k+1)} = \max(0, p_n)$$
>
> where normal reaction force magnitude is $\lambda_n = \rho_c \max(0, -p_n)$.
>
> #### Tangential Step (Coulomb Friction):
>
>$$\mathbf{z}_t^{(k+1)} = \max\left(0, 1 - \frac{\mu_c \lambda_n}{\rho_c \Vert{}\mathbf{p}_t\Vert{}_2}\right) \mathbf{p}_t$$
>
>#### Combined Update:
>
>$$\mathbf{z}_{c,c}^{(k+1)} = z_n^{(k+1)} \mathbf{n}_c + \mathbf{z}_t^{(k+1)}$$
>

### Global Step with Contact ($\mathbf{x}$-update)

Differentiating the combined elasticity and contact Augmented Lagrangian $\mathcal{L}_{\boldsymbol{\rho}}$ with respect to global positions $\mathbf{x}$ and setting the gradient to zero:

$$\nabla_{\mathbf{x}} \left( \frac{1}{2\Delta t^2} (\mathbf{x} - \mathbf{y})^T \mathbf{M} (\mathbf{x} - \mathbf{y}) + \frac{\rho_s}{2} (\mathbf{S}\mathbf{x} - \mathbf{z}_s^{(k)} + \mathbf{u}_s^{(k)})^T (\mathbf{S}\mathbf{x} - \mathbf{z}_s^{(k)} + \mathbf{u}_s^{(k)}) + \frac{\rho_c}{2} (\mathbf{C}\mathbf{x} - \mathbf{z}_c^{(k)} + \mathbf{u}_c^{(k)})^T (\mathbf{C}\mathbf{x} - \mathbf{z}_c^{(k)} + \mathbf{u}_c^{(k)}) \right) = \mathbf{0}$$

This yields a sparse linear system:

$$\underbrace{\left( \frac{\mathbf{M}}{\Delta t^2} + \rho_s \mathbf{S}^T \mathbf{S} + \rho_c \mathbf{C}^T \mathbf{C} \right)}_\mathbf{H} \mathbf{x}^{(k+1)} = \frac{\mathbf{M}}{\Delta t^2} \mathbf{y} + \rho_s \mathbf{S}^T \left( \mathbf{z}_s^{(k)} - \mathbf{u}_s^{(k)} \right) + \rho_c \mathbf{C}^T \left( \mathbf{z}_c^{(k)} - \mathbf{u}_c^{(k)} \right)$$

Unlike the pure elasticity Hessian ($\frac{\mathbf{M}}{\Delta t^2} + \rho_s \mathbf{S}^T \mathbf{S}$), the term $\rho_c \mathbf{C}^T \mathbf{C}$ depends on active contact pairs. Because contacts form, slip, and break dynamically, $\mathbf{C}$ changes topology every time step, destroying the static structure of the system matrix $\mathbf{H}$, preventing pre-factorization.

#### Primal Constraint Approximation

Recall the canonical ADMM split optimization framework:

$$\min_{\mathbf{x}, \mathbf{z}} \; f(\mathbf{x}) + g(\mathbf{z}) \quad \text{s.t.} \quad \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} = \mathbf{c}$$

For our coupled elasticity and contact problem, we split internal elasticity forces ($\mathbf{S}$) and contact kinematics ($\mathbf{C}$) into separate linear constraints:

$$\mathbf{S}\mathbf{x} - \mathbf{z}_s = \mathbf{0} \quad \text{and} \quad \mathbf{C}\mathbf{x} - \mathbf{z}_c = \mathbf{0}$$

As the algorithm converges ($k \to \infty$), the primal contact residual vanishes:

$$\lim_{k \to \infty} \left( \mathbf{C}\mathbf{x}^{(k+1)} - \mathbf{z}_c^{(k)} \right) = \mathbf{0} \quad \implies \quad \mathbf{C}\mathbf{x}^{(k+1)} \approx \mathbf{z}_c^{(k)}$$

In the exact stationarity condition for $\mathbf{x}$, the contact term appears as $\rho_c \mathbf{C}^T \left( \mathbf{C}\mathbf{x} - \mathbf{z}_c^{(k)} + \mathbf{u}_c^{(k)} \right)$. Using the convergent limit $\mathbf{C}\mathbf{x} \approx \mathbf{z}_c^{(k)}$, we evaluate this term without implicitly expanding $\mathbf{C}^T \mathbf{C}$ [G. Daviet 2023](https://research.nvidia.com/labs/prl/admm_hair/):

$$\rho_c \mathbf{C}^T \left( \mathbf{C}\mathbf{x} - \mathbf{z}_c^{(k)} + \mathbf{u}_c^{(k)} \right) \xrightarrow{\mathbf{C}\mathbf{x} \to \mathbf{z}_c^{(k)}} \rho_c \mathbf{C}^T \left( \underbrace{\mathbf{z}_c^{(k)} - \mathbf{z}_c^{(k)}}_{= \mathbf{0}} + \mathbf{u}_c^{(k)} \right) = \rho_c \mathbf{C}^T \mathbf{u}_c^{(k)}$$

Substituting this back into the stationarity equation yields:

$$\frac{\mathbf{M}}{\Delta t^2}(\mathbf{x} - \mathbf{y}) + \rho_s \mathbf{S}^T \left( \mathbf{S}\mathbf{x} - \mathbf{z}_s^{(k)} + \mathbf{u}_s^{(k)} \right) + \rho_c \mathbf{C}^T \mathbf{u}_c^{(k)} = \mathbf{0}$$

Moving all non-$\mathbf{x}$ terms to the right-hand side isolates $\mathbf{x}^{(k+1)}$ in the final decoupled sparse linear system:

$$\underbrace{\left( \frac{\mathbf{M}}{\Delta t^2} + \rho_s \mathbf{S}^T \mathbf{S} \right)}_{\mathbf{H}} \mathbf{x}^{(k+1)} = \frac{\mathbf{M}}{\Delta t^2} \mathbf{y} + \rho_s \mathbf{S}^T \left( \mathbf{z}_s^{(k)} - \mathbf{u}_s^{(k)} \right) - \rho_c \mathbf{C}^T \mathbf{u}_c^{(k)}$$

Note that $\mathbf{C}\mathbf{x}^{(k+1)} \approx \mathbf{z}_c^{(k)}$ is a form of primal linearization which means contact stiffness relies entirely on dual ascent.
