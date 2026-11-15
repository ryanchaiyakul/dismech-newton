"""Centralized physical, numerical, solver, contact, collision, and adjoint constants plus shared Warp types."""

import warp as wp

# -- Model namespace & carried state --------------------------------------------------------------

NAMESPACE = "dismech"
ATTR_PREFIX = f"{NAMESPACE}:"
ROD_GROUP = 0  # every colliding rod capsule's collision group: Newton pairs group 0 with nothing (the solver's own
# contact finds the rod's), so no rigid solver or coupler collides with the capsules
CARRIED = ("edge_d1_q", "triplet_ref_twist_q", "triplet_mem_q", "face_mem_q")  # carried beyond DOFs and strains

# -- Shared Warp vector and matrix typedefs -------------------------------------------------------

vec5f = wp.types.vector(5, float)  # a triplet's strains [eps_e, eps_f, kappa1, kappa2, tau]
vec6f = wp.types.vector(6, float)  # a face's strains (membrane 3, bending 3)
vec8f = wp.types.vector(8, float)
vec10f = wp.types.vector(10, float)  # [stiffness, damping] per triplet strain
vec11f = wp.types.vector(11, float)
vec12f = wp.types.vector(12, float)  # [stiffness, damping] per face strain
vec21f = wp.types.vector(21, float)

vec5i = wp.types.vector(5, wp.int32)
vec9i = wp.types.vector(9, wp.int32)
vec11i = wp.types.vector(11, wp.int32)
vec21i = wp.types.vector(21, wp.int32)

mat23f = wp.types.matrix((2, 3), float)
mat63f = wp.types.matrix((6, 3), float)
mat5x11f = wp.types.matrix((5, 11), float)
mat6x21f = wp.types.matrix((6, 21), float)

# -- NVRTC module option presets ------------------------------------------------------------------

FWD_MODULE_OPTS = {"enable_backward": False}
ADJ_MODULE_OPTS = {"enable_backward": True, "max_unroll": 32, "optimization_level": 2}

# -- Execution, implicit solver, line search, PCG & cuDSS -----------------------------------------

BLOCK = 256  # threads per block of a kernel that sums over its threads (tile reductions, wp.launch's default)
ARMIJO_C1 = 1.0e-4  # Armijo's sufficient decrease
OBJECTIVE_NOISE = 1.0e-5  # a rise of the objective within its float32 rounding
PCG_SWEEPS = 2  # SPD PCG sweeps on two-sided contacts (m = 0 scales A^-1 b; m = 1..CG-1 add CG-1 back-solves)
LINESEARCH = 4  # step lengths tried per iteration, halving from 1
NEST = 8  # iterations nested per conditional chain (:meth:`SolverDiSMech.step`)
# the barrier's step has converged when its Newton step moves no node more than this many radii, nor changes a pair's
# force by more than this fraction of (the force that moves its mass a radius in a step + the force it holds)
TOLERANCE = 0.01
ULP = 1.0e-6  # float32's reach: a few of its epsilons of a coordinate (+ the cap): forces below it, in a
# DOF's inertia, are not resolved (the force balance's floor)
EPS32 = 1.1920929e-07  # float32's epsilon: a coordinate's ulp is at most this times it

CUDA_R_32F = 0  # cudaDataType codes
CUDA_R_64F = 1
CUDA_R_32I = 10

# -- OGC barrier & friction -----------------------------------------------------------------------

SMOOTH = wp.constant(0.02)  # the barrier's activation margin above the thickness (in thicknesses): a pair resting at
# its thickness already has curvature, so A sees it
LOAD = wp.constant(0.25)  # a node's pairs are stiffened until they hold its load at most this many thicknesses deep
GROW = wp.constant(2.0)  # a node's load scale grows at most this factor per step
DECAY = wp.constant(0.99)  # and relaxes by this factor per step, toward 1
MAX_SCALE = wp.constant(1.0e5)  # and stays below this (unbounded, it spreads pair to pair and overflows k)
CURVATURE = wp.constant(1.0e2)  # A holds a pair's normal curvature to at most this many k (its log's: d >= tau/10)
STIFFNESS = 1.0e3  # each pair's barrier stiffness, in units of its inertia 1 / (dt^2 C M^-1 C^T)
SIDES = 16  # a shape's pair keys per feature: a line contact's ends, or a pad's points (collide)
PAD_FLOOR = wp.constant(0.05)  # A holds a pad pair at least at this fraction of its depth (:func:`_majorizer`)
SLIP_SPEED = 0.01  # friction is smoothed below this slip speed [m/s]
LIFT = wp.constant(0.25)  # a shape pair's least gap at a step's start, from the barrier's zero (in thicknesses)
IMPACT = wp.constant(0.8)  # the line search's longest step: this fraction of the first pair's impact (IPC's)
NONE = wp.constant(wp.int64(9223372036854775807))  # an empty place's sort key

# -- Collision detection --------------------------------------------------------------------------

PAIRS = 8  # the pair list's places per feature (node or edge), shared: the scenes' peak is 1.7 (cloth_style3d)
PAD_DEPTH = wp.constant(0.5)  # a pad holds the outer half of a pair's thickness, its core's barrier the rest
REACH = 2.0  # pairs within this many thicknesses: the thickness, plus two move caps (a radius each)
EXCLUSION = 1.5  # rod-rod pairs closer than this many thicknesses along one rod are not contacts
OPEN = 1.0e-4  # an edge-edge closest point this close to an end (fraction of the edge) is the node's: blocks are open
PARALLEL = 1.0e-3  # edges with sin^2 of their angle below this: no edge-edge pair (their nodes' pairs hold it, IPC)
BISECT = 14  # bisections for a rod edge's closest point to a shape
FACE_PASSES = 8  # alternating closest points for a triangle's point nearest a shape
INSIDE = 0.01  # its weights at least this (else on its edges, theirs)
NEARER = 0.01  # an edge's or triangle's shape pair: closer than its nodes by this many thicknesses (else theirs)
FLAT = 0.015  # 1 - cos 10 deg: normals along an edge within this of its closest point's are its face's (a line contact)

# -- Adjoint & Dahl hysteresis --------------------------------------------------------------------

CG_TOLERANCE = 1.0e-8  # the contact solve's relative residual, in the preconditioner's norm (exact but for rod-rod pairs)
CG_ITERATIONS = 100
DAHL_DEADBAND = wp.constant(1.0e-6)  # a smaller strain change keeps sigma's branch: noise cannot erode sigma
