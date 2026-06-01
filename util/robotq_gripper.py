# robotiq_gripper.py — minimal version of the standard community module
import socket, threading, time

class RobotiqGripper:
    OPEN, CLOSED = 0, 255

    def __init__(self):
        self.socket = None
        self.lock = threading.Lock()

    def connect(self, hostname, port=63352, timeout=2.0):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        self.socket.connect((hostname, port))

    def disconnect(self):
        self.socket.close()

    def _cmd(self, cmd):
        with self.lock:
            self.socket.sendall((cmd + "\n").encode())
            return self.socket.recv(1024).decode().strip()

    def _set(self, var, val):
        return self._cmd(f"SET {var} {val}")
    
    def _get(self, var):
        return self._cmd(f"GET {var}").split()[1]

    def activate(self):
        if self._get("STA") != "3":
            self._set("ACT", 1)
            while self._get("STA") != "3":
                time.sleep(0.1)

    def move(self, pos, speed=255, force=255):
        self._set("POS", max(0, min(255, pos)))
        self._set("SPE", max(0, min(255, speed)))
        self._set("FOR", max(0, min(255, force)))
        self._set("GTO", 1)

    def move_and_wait(self, pos, speed=255, force=255):
        self.move(pos, speed, force)
        while self._get("PRE") != str(pos):
            time.sleep(0.05)
        while self._get("OBJ") == "0":
            time.sleep(0.05)
        return int(self._get("POS")), int(self._get("OBJ"))

    def position(self):
        return int(self._get("POS"))