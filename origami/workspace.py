"""The `Workspace` class ties the whole rig together.

A workspace owns the two arms, the current paper state, the magnet registry and
the board dimensions.  High-level choreography in `origami.actions` operates
on a ``Workspace``; the demos build one and run a fold recipe against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .arm import Arm, ArmConfig
from .magnets import MagnetRegistry
from .paper import Paper


@dataclass
class Workspace:
    """Container for both arms plus the paper, magnets and board geometry.

    Parameters
    ----------
    left, right : origami.arm.Arm
        The two robot arms.
    paper : origami.paper.Paper
        Current paper state.
    magnets : origami.magnets.MagnetRegistry, optional
        Registry of magnet holders.  Defaults to an empty registry.
    board_width : float, optional
        Board width along ``+x`` (metres).  Defaults to ``config.BOARD_WIDTH``.
    board_height : float, optional
        Board height along ``+y`` (metres).  Defaults to
        ``config.BOARD_HEIGHT``.

    See Also
    --------
    Workspace.simulated, Workspace.hardware : Convenience constructors.
    """

    left: Arm
    right: Arm
    paper: Paper
    magnets: MagnetRegistry = field(default_factory=MagnetRegistry)
    board_width: float = config.BOARD_WIDTH
    board_height: float = config.BOARD_HEIGHT

    def arm(self, side: str) -> Arm:
        """Return the arm for a given side.

        Parameters
        ----------
        side : {'left', 'l', 'right', 'r'}
            Which arm to return (case-insensitive).

        Returns
        -------
        origami.arm.Arm

        Raises
        ------
        ValueError
            If ``side`` is not recognised.
        """
        key = side.lower()
        if key in ("l", "left"):
            return self.left
        if key in ("r", "right"):
            return self.right
        raise ValueError(f"unknown arm side: {side!r}")

    def other_arm(self, side: str) -> Arm:
        """Return the arm *opposite* to the given side.

        Parameters
        ----------
        side : {'left', 'l', 'right', 'r'}

        Returns
        -------
        origami.arm.Arm
        """
        return self.right if self.arm(side) is self.left else self.left

    # ------------------------------------------------------------------ #
    # Builders
    # ------------------------------------------------------------------ #
    @classmethod
    def simulated(cls, paper: Paper | None = None, magnets: MagnetRegistry | None = None,
                  arm_config: ArmConfig | None = None) -> "Workspace":
        """Build a fully simulated workspace (no hardware required).

        Parameters
        ----------
        paper : origami.paper.Paper or None, optional
            Initial paper; defaults to a sheet at the board's bottom-left corner.
        magnets : origami.magnets.MagnetRegistry or None, optional
            Magnet registry; defaults to empty.
        arm_config : origami.arm.ArmConfig or None, optional
            Shared motion defaults for both arms.

        Returns
        -------
        Workspace
        """
        left = Arm.simulated("left", config.left_calibration(), arm_config)
        right = Arm.simulated("right", config.right_calibration(), arm_config)
        paper = paper or Paper.rectangle(config.PAPER_WIDTH, config.PAPER_HEIGHT, origin=(0.0, 0.0))
        return cls(left, right, paper, magnets or MagnetRegistry())

    @classmethod
    def hardware(cls, paper: Paper | None = None, magnets: MagnetRegistry | None = None,
                 arm_config: ArmConfig | None = None) -> "Workspace":  # pragma: no cover - needs robots
        """Build a live workspace driving the real arms and gripper.

        Parameters
        ----------
        paper : origami.paper.Paper or None, optional
            Initial paper; defaults to a sheet at the board's bottom-left corner.
        magnets : origami.magnets.MagnetRegistry or None, optional
            Magnet registry; defaults to empty.
        arm_config : origami.arm.ArmConfig or None, optional
            Shared motion defaults for both arms.

        Returns
        -------
        Workspace
        """
        left = Arm.real("left", config.LEFT_ARM_IP, config.left_calibration(), config=arm_config)
        right = Arm.real("right", config.RIGHT_ARM_IP, config.right_calibration(),
                         gripper_ip=config.GRIPPER_IP, gripper_port=config.GRIPPER_PORT,
                         config=arm_config)
        paper = paper or Paper.rectangle(config.PAPER_WIDTH, config.PAPER_HEIGHT, origin=(0.0, 0.0))
        return cls(left, right, paper, magnets or MagnetRegistry())


