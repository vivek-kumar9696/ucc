# Construct a custom compiler
import os

try:
    from qiskit.utils.parallel import default_num_processes
except ImportError:
    # Qiskit 1.0.0 doesn't have this function, so we make it ourselves
    from qiskit.utils.parallel import CPU_COUNT

    def default_num_processes():
        return CPU_COUNT


from qiskit.transpiler import PassManager
from qiskit import user_config
from qiskit.transpiler import Target
from qiskit.transpiler.passes import (
    ApplyLayout,
    ConsolidateBlocks,
    CollectCliffords,
    HighLevelSynthesis,
    HLSConfig,
    SabreLayout,
    SabreSwap,
    VF2Layout,
    CommutativeCancellation,
    Collect2qBlocks,
    UnitarySynthesis,
    Optimize1qGatesDecomposition,
    VF2PostLayout,
)

from routing_algos.FDLSswap import FDLSSwap
from typing import Optional


CONFIG = user_config.get_config()


class UCCDefault1:
    def __init__(
        self, local_iterations: int = 1, target_device: Optional[Target] = None
    ):
        """
        Create a new instance of UCCDefault1 compiler

            Args:
                local_iterations (int): Number of times to run the local passes
                target_device (qiskit.transpiler.Target): (Optional) The target device to compile the circuit for
        """
        self.pass_manager = PassManager()
        self._1q_basis = ["rz", "rx", "ry", "h"]
        self._2q_basis = ["cx"]
        self.target_basis = self._1q_basis + self._2q_basis

        self.special_commutations = {
            ("rx", "cx"): {
                (0,): False,
                (1,): True,
            },
            ("rz", "cx"): {
                (0,): True,
                (1,): False,
            },
        }
        self._add_local_passes(local_iterations)
        self._add_map_passes(target_device)

    @property
    def default_passes(self):
        return

    def _add_local_passes(self, local_iterations):
        for _ in range(local_iterations):
            self.pass_manager.append(Optimize1qGatesDecomposition())
            self.pass_manager.append(CommutativeCancellation())
            self.pass_manager.append(Collect2qBlocks())
            self.pass_manager.append(ConsolidateBlocks(force_consolidate=True))
            self.pass_manager.append(
                UnitarySynthesis(basis_gates=self.target_basis)
            )
            # self.pass_manager.append(Optimize1qGatesDecomposition(basis=self._1q_basis))
            self.pass_manager.append(CollectCliffords())
            self.pass_manager.append(
                HighLevelSynthesis(hls_config=HLSConfig(clifford=["greedy"]))
            )

            # Add following passes if merging single qubit rotations that are interrupted by a commuting 2 qubit gate is desired
            # self.pass_manager.append(Optimize1qGatesSimpleCommutation(basis=self._1q_basis))
            # self.pass_manager.append(BasisTranslator(sel, target_basis=self.target_basis))

    def _add_map_passes(self, target_device: Optional[Target] = None):
        if target_device is not None:
            coupling_map = target_device.build_coupling_map()
            # self.pass_manager.append(ElidePermutations())
            # self.pass_manager.append(SpectralMapping(coupling_list))
            # self.pass_manager.append(SetLayout(pass_manager_config.initial_layout))
            self.pass_manager.append(
                SabreLayout(
                    coupling_map,
                    seed=1,
                    max_iterations=4,
                    swap_trials=_get_trial_count(20),
                    layout_trials=_get_trial_count(20),
                )
            )

            self.pass_manager.append(VF2Layout(target=target_device))
            self.pass_manager.append(ApplyLayout())
            self.pass_manager.append(
                SabreSwap(
                    coupling_map,
                    heuristic="decay",
                    seed=1,
                    trials=_get_trial_count(20),
                )
            )
            # self.pass_manager.append(MapomaticLayout(coupling_map))
            self.pass_manager.append(VF2PostLayout(target=target_device))
            self.pass_manager.append(ApplyLayout())
            self._add_local_passes(1)
            self.pass_manager.append(VF2PostLayout(target=target_device))
            self.pass_manager.append(ApplyLayout())

    def run(self, circuits):
        return self.pass_manager.run(circuits)

class UCCfdls(UCCDefault1):
    def __init__(
        self, local_iterations: int = 1, target_device: Optional[Target] = None
    ):
        """
        Create a new instance of UCCDefault1 compiler

            Args:
                local_iterations (int): Number of times to run the local passes
                target_device (qiskit.transpiler.Target): (Optional) The target device to compile the circuit for
        """

        super().__init__(local_iterations=local_iterations, target_device=target_device)
        

    @property
    def default_passes(self):
        return


    def _add_map_passes(self, target_device: Optional[Target] = None):
        if target_device is not None:
            coupling_map = target_device.build_coupling_map()

            # --- 1 · Choose a (good-enough) initial placement -----------------
            self.pass_manager.append(
                SabreLayout(
                    coupling_map,
                    seed=1,
                    max_iterations=4,
                    swap_trials=_get_trial_count(20),
                    layout_trials=_get_trial_count(20),
                )
            )
            self.pass_manager.append(VF2Layout(target=target_device))
            self.pass_manager.append(ApplyLayout())

            # --- 2 · Route with Filtered Depth-Limited Search -----------------
            #     (replaces the old SabreSwap pass)
            self.pass_manager.append(
                FDLSSwap(
                    coupling_map,
                    depth_limit=3,        # k  – max swaps per search
                    lookahead_layers=2,   # h  – Qᵢ filter horizon
                    ds_discount=0.95,     # λ  – distance-metric discount
                    seed=1,               # deterministic tie-breaks
                )
            )

            # --- 3 · Optional post-layout clean-ups ---------------------------
            self.pass_manager.append(VF2PostLayout(target=target_device))
            self.pass_manager.append(ApplyLayout())

            # Any custom local optimisations you already had
            self._add_local_passes(1)

            # Final VF2 pass to tighten things even more
            self.pass_manager.append(VF2PostLayout(target=target_device))
            self.pass_manager.append(ApplyLayout())

    def run(self, circuits):
        return self.pass_manager.run(circuits)

def _get_trial_count(default_trials=5):
    if CONFIG.get("sabre_all_threads", None) or os.getenv(
        "QISKIT_SABRE_ALL_THREADS"
    ):
        return default_num_processes()
    return default_trials
