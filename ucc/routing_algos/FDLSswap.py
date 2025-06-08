"""Routing via SWAP insertion using Filtered Depth-Limited Search (FDLS)."""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterable, Tuple, List, Set

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.coupling import CouplingMap


logger = logging.getLogger(__name__)


class FDLSSwap(TransformationPass):
    r"""
    Map a circuit onto a backend topology by inserting SWAPs chosen with the
    **Filtered Depth-Limited Search** strategy from:

        *Sanjiang Li, Xiangzhen Zhou, Yuan Feng –
        "Qubit Mapping Based on Subgraph Isomorphism and
        Filtered Depth-Limited Search", 2021.*
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        lookahead_layers: int = 2,
        depth_limit: int = 3,
        ds_discount: float = 0.95,
        seed: int | None = None,
    ):
        super().__init__()
        self.coupling_map = coupling_map
        self.graph = coupling_map.graph
        self.dist = coupling_map.distance_matrix
        self.k = depth_limit
        self.h = lookahead_layers
        self.lmbd = ds_discount
        self.seed = seed
        self._swap_gate = SwapGate()

    # ---------------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------------
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if self.coupling_map is None:
            raise TranspilerError("FDLSSwap requires a CouplingMap.")

        if len(dag.qregs) != 1 or "q" not in dag.qregs:
            raise TranspilerError("FDLSSwap expects a physical circuit with a single qreg.")

        canonical_qreg = dag.qregs["q"]
        self._canonical_qreg = canonical_qreg
        current_layout = Layout.generate_trivial_layout(canonical_qreg)  # physical == virtual idx
        out_dag = dag.copy_empty_like()

        front_layer = deque(dag.front_layer())

        while front_layer or dag.op_nodes():
            # --------------------------------------------------------------
            # 1.  Execute any front-layer gates that are already satisfied.
            # --------------------------------------------------------------
            executed_any = False
            executable_nodes = [
                n for n in list(front_layer)
                    if isinstance(n, DAGOpNode)                  # ← new guard
                        and self._gate_executable(n, current_layout)
]
            for node in executable_nodes:
                self._apply_physical_op(out_dag, node, current_layout)
                front_layer.remove(node)
                for succ in dag.successors(node):
                    if isinstance(succ, DAGOpNode) and succ not in front_layer:
                        front_layer.append(succ)
                dag.remove_op_node(node)
                executed_any = True

            if executed_any:
                continue

            # --------------------------------------------------------------
            # 2.  Need SWAPs – run FDLS to pick the best short sequence.
            # --------------------------------------------------------------
            swap_seq = self._best_fdls_swap_sequence(
                current_layout, front_layer, depth_limit=self.k, lookahead=self.h
            )
            if not swap_seq:
                raise TranspilerError(
                    "FDLS could not find any swap sequence – are device and circuit sizes compatible?"
                )

            for a, b in swap_seq:
                # Update layout *and* write SWAP to output DAG.
                current_layout.swap(a, b)
                out_dag.apply_operation_back(
                    self._swap_gate, (canonical_qreg[a], canonical_qreg[b]), (), check=False
                )

        return out_dag

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _gate_executable(self, node: DAGOpNode, layout: Layout) -> bool:
        """Return True iff the two-qubit gate `node` can be run on current mapping."""
        if len(node.qargs) != 2:
            return True  # single-qubit directives never need routing
        q0, q1 = node.qargs
        p0 = layout[q0]
        p1 = layout[q1]
        return self.coupling_map.graph.has_edge(p0, p1)

    def _apply_physical_op(self, out_dag, node, layout):
        """Copy `node` to `out_dag` with physical qubits substituted."""
        phys_qargs = []
        for virt_q in node.qargs:
            phys_idx = layout[virt_q]          # int
            try:
                phys_q = self._canonical_qreg[phys_idx]  # Qubit
            except IndexError:                  # just in case
                raise TranspilerError(
                    f"Layout maps to idx {phys_idx}, but register has size {len(self._canonical_qreg)}"
                )
            phys_qargs.append(phys_q)
        #new_node = DAGOpNode.from_instruction(node.op, qargs=phys_qargs, cargs=node.cargs)
        out_dag.apply_operation_back(node.op, phys_qargs, node.cargs, check=False)

    # ---------------------------------------------------------------------
    # FDLS core
    # ---------------------------------------------------------------------
    def _best_fdls_swap_sequence(
        self,
        layout: Layout,
        front_layer: Iterable[DAGOpNode],
        depth_limit: int,
        lookahead: int,
    ) -> List[Tuple[int, int]]:
        """
        Return a (possibly empty) **sequence** of ≤ `depth_limit` SWAPs that maximises

            g_val = (# front-layer gates enabled) / (3 * num_swaps)

        when applied to `layout`, subject to the **Q₀/Qᵢ/Dₛ filters**.
        """
        # Pre-compute front-layer logical qubits
        front_qubits = {q for n in front_layer for q in n.qargs}
        logical_front = [layout[q] for q in front_qubits]

        best_score = -1.0
        best_swap_seq: List[Tuple[int, int]] = []

        # The search tree nodes are tuples (partial_seq, partial_layout)
        stack: List[Tuple[List[Tuple[int, int]], Layout]] = [
            ([], layout.copy())
        ]  # start with empty sequence

        while stack:
            seq, seq_layout = stack.pop()

            # ----- Evaluate current candidate --------------------------------
            if seq:  # non-empty sequences only
                enabled = sum(
                    self._gate_executable(n, seq_layout) for n in front_layer
                )
                score = enabled / (3 * len(seq))
                if score > best_score:
                    best_score = score
                    best_swap_seq = seq

            # Depth limit reached?
            if len(seq) == depth_limit:
                continue

            # ----- Generate children (filtered) ------------------------------
            candidate_swaps = self._generate_filtered_swaps(
                logical_front, seq_layout, front_layer, level=len(seq), lookahead=lookahead
            )
            for (a, b) in candidate_swaps:
                new_layout = seq_layout.copy()
                new_layout.swap(a, b)
                stack.append((seq + [(a, b)], new_layout))

        return best_swap_seq

    # .........................................................................
    #  Filters:  Q₀ (must touch a front-layer qubit),
    #            Qᵢ (look-ahead layers),
    #            Dₛ (discounted distance metric must not increase)
    # .........................................................................
    def _generate_filtered_swaps(
        self,
        logical_front: List[int],
        layout: Layout,
        front_layer: Iterable[DAGOpNode],
        level: int,
        lookahead: int,
    ) -> Set[Tuple[int, int]]:
        """Return SWAPs that satisfy Q₀, Qᵢ, and Dₛ filters."""
        swap_set: Set[Tuple[int, int]] = set()

        # ----- Q₀ filter: touch at least one qubit in logical_front ----------
        touch = set(logical_front)

        # ----- Qᵢ filter: include look-ahead layers --------------------------
        if lookahead > 0:
            la_qubits = self._collect_lookahead_qubits(front_layer, depth=lookahead)
            touch.update(layout[q] for q in la_qubits)

        # Iterate over all edges of coupling graph that touch those qubits
        for p in touch:
            for nbr in self.graph.neighbors(p):
                if (p, nbr) in swap_set or (nbr, p) in swap_set:
                    continue

                # Dₛ filter: check discounted distance metric
                if not self._pass_ds_filter(layout, p, nbr, front_layer):
                    continue

                swap_set.add((p, nbr))

        return swap_set

    # .........................................................................

    def _collect_lookahead_qubits(self, front_layer, depth):
        """Return logical qubits up to `depth` successive layers ahead."""
        qubits = set()
        frontier = deque((n, 0) for n in front_layer)
        seen = set(front_layer)
        while frontier:
            node, d = frontier.popleft()
            if d >= depth:
                continue
            for succ in node.q_successors():
                if succ in seen:
                    continue
                seen.add(succ)
                qubits.update(succ.qargs)
                frontier.append((succ, d + 1))
        return qubits

    def _pass_ds_filter(self, layout, p, nbr, front_layer):
        """
        Discounted-distance filter Dₛ: Keep the swap (p,nbr) iff it **does not
        increase** Σ (λ^ℓ · dist) over the first s=|front_layer| layers.

        Here we approximate with the current front only (cheaper but effective).
        """
        def total_distance(lay):
            dist = 0
            for node in front_layer:
                q0, q1 = node.qargs
                dist += self.dist[lay[q0]][lay[q1]]
            return dist

        before = total_distance(layout)
        after_layout = layout.copy()
        after_layout.swap(p, nbr)
        after = total_distance(after_layout)

        return after <= before * self.lmbd


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------
#
# from qiskit import QuantumCircuit, transpile
# from qiskit.transpiler import PassManager
# from qiskit.transpiler.passes import SetLayout
# from qiskit.transpiler.coupling import CouplingMap
#
# circ = QuantumCircuit(5)
# circ.cx(0, 1)
# circ.cx(2, 3)
# circ.cx(3, 4)
# circ.cx(1, 2)
#
# cmap = CouplingMap.from_line(5)
# passmanager = PassManager([
#     SetLayout(Layout.generate_trivial_layout(circ.qregs["q"])),
#     FDLSSwap(cmap)
# ])
# mapped = passmanager.run(circ)
# print(mapped)
