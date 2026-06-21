# robotiq_gripper.py — minimal version of the standard community module
import socket, threading, time
from enum import Enum
class RobotiqGripper:
    """
    Communicates with the gripper directly, via socket with string commands, leveraging string names for variables.
    """
    # WRITE VARIABLES (CAN ALSO READ)
    ACT = 'ACT'  # act : activate (1 while activated, can be reset to clear fault status)
    GTO = 'GTO'  # gto : go to (will perform go to with the actions set in pos, for, spe)
    ATR = 'ATR'  # atr : auto-release (emergency slow move)
    ADR = 'ADR'  # adr : auto-release direction (open(1) or close(0) during auto-release)
    FOR = 'FOR'  # for : force (0-255)
    SPE = 'SPE'  # spe : speed (0-255)
    POS = 'POS'  # pos : position (0-255), 0 = open
    # READ VARIABLES
    STA = 'STA'  # status (0 = is reset, 1 = activating, 3 = active)
    PRE = 'PRE'  # position request (echo of last commanded position)
    OBJ = 'OBJ'  # object detection (0 = moving, 1 = outer grip, 2 = inner grip, 3 = no object at rest)
    FLT = 'FLT'  # fault (0=ok, see manual for errors if not zero)

    ENCODING = 'UTF-8'  # ASCII and UTF-8 both seem to work

    class GripperStatus(Enum):
        """Gripper status reported by the gripper. The integer values have to match what the gripper sends."""
        RESET = 0
        ACTIVATING = 1
        # UNUSED = 2  # This value is currently not used by the gripper firmware
        ACTIVE = 3

    class ObjectStatus(Enum):
        """Object status reported by the gripper. The integer values have to match what the gripper sends."""
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self):
        """Constructor."""
        self.socket = None
        self.command_lock = threading.Lock()
        self._open_position = 0
        self._close_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255


    def connect(self, hostname: str, port: int = 63352, timeout: float = 2.0) -> None:
        """Connects to a gripper at the given address.
        Parameters
        hostname : str
            The hostname or IP address of the gripper's socket server.
        port : int, optional
            The TCP port of the gripper's socket server. Default is 63352.
        timeout : float, optional
            The timeout in seconds for the socket connection. Default is 2.0 seconds.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        self.socket.connect((hostname, port))

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        self.socket.close()

    def _send_cmd(self, cmd: str) -> str:
        """Send a command to the gripper and returns the response. 
        Parameters
        ----------
        cmd : str
            The command to send to the gripper.

        Returns
        -------
        str
            The response from the gripper.
        """
        with self.command_lock:
            self.socket.sendall((cmd + "\n").encode())
            data = self.socket.recv(1024)
            data_formatted = data.decode(self.ENCODING).strip()
            return data_formatted

    def _got_ack(self, response: str) -> bool:
        """Check if the response from the gripper after sending a command is an acknowledgment (ACK)."""
        return response.lower() == "ack"

    def _set_var(self, var, val):
        """Set a variable on the gripper. 
        See Write Variables above for the available variables.
        Parameters
        ----------
        var : str
            The variable name to set on the gripper. Should be passed in via the class/self constants (e.g., self.POS / RobotiqGripper.POS, self.FOR / RobotiqGripper.FOR, etc.).
        val : int
            The value to set for the variable.
        Returns
        -------
        str
            The response from the gripper after setting the variable.
        Raises
        ------
        RuntimeError
            If the gripper does not acknowledge the command to set the variable.
        """
        response = self._send_cmd(f"SET {var} {val}")
        if not self._got_ack(response):
            raise RuntimeError(f"Failed to set variable {var} to {val}. Response: {response}")
        return response
    
    def _get_var(self, var) -> int:
        """Get a variable from the gripper.
        See Read (and Write!) Variables above for the available variables.
        Parameters
        ----------
        var : str
            The variable name to get from the gripper.

        Returns
        -------
        int
            The value of the variable.

        Raises
        ------
        ValueError
            If the response from the gripper does not match the requested variable.
        """

        # Note: the gripper's response to a GET command is of the form "VAR value",  where VAR is an echo of the variable name, and 
        # value is the returned value. 
        # note some special variables (like FLT) may send 2 bytes, instead of an integer. We assume integer here for simplicity, but this may need to be adapted if we want to read those variables.
        data = self._send_cmd(f"GET {var}")
        var_name, value_str = data.split()
        if var_name != var:
            raise ValueError(f"Unexpected response {data}: does not match '{var}'")
        value = int(value_str)  # we assume the value is an integer, which is the case for all variables we currently use. This may need to be adapted if we want to read variables that return non-integer values.
        return value

    def activate(self):
        if self._get_var(self.STA) != 3:
            self._set_var(self.ACT, 1)
            while self._get_var(self.STA) != 3:
                time.sleep(0.1)

    def move(self, pos, speed=255, force=255):
        self._set_var(self.POS, max(0, min(255, pos)))
        self._set_var(self.SPE, max(0, min(255, speed)))
        self._set_var(self.FOR, max(0, min(255, force)))
        self._set_var(self.GTO, 1)

    def move_and_wait(self, pos, speed=255, force=255):
        self.move(pos, speed, force)
        while self._get_var(self.PRE) != pos:
            #print(f"Waiting for gripper to reach position {pos}, current position: {self._get_var(self.PRE)}, sleeping for 1 second...")
            time.sleep(1)
        return

    def position(self):
        return self._get_var(self.POS)
    
    def open(self, blocking=True):
        if blocking:
            self.move_and_wait(self._open_position)
        else:
            self.move(self._open_position)

    def close(self, blocking=True):
        if blocking:
            self.move_and_wait(self._close_position)
        else:
            self.move(self._close_position)

    def is_active(self) -> bool:
        """Returns whether the gripper is active."""
        status = self._get_var(self.STA)
        return RobotiqGripper.GripperStatus(status) == RobotiqGripper.GripperStatus.ACTIVE

    def get_current_position(self) -> int:
        """Returns the current position (0-255) of the gripper as returned by the physical hardware."""
        return self._get_var(self.POS)
    
    def is_open(self) -> bool:
        """Returns whether the current position is considered as being fully open."""
        return self.get_current_position() <= self.get_open_position()

    def is_closed(self) -> bool:
        """Returns whether the current position is considered as being fully closed."""
        return self.get_current_position() >= self.get_closed_position()