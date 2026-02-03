import atexit
from collections.abc import Sequence
import logging
from threading import Event
from threading import Lock
from threading import Thread
import time

from dynamixel_sdk import COMM_SUCCESS
from dynamixel_sdk import GroupSyncRead
from dynamixel_sdk import GroupSyncWrite
from dynamixel_sdk import PortHandler
from dynamixel_sdk import Protocol2PacketHandler
from dynamixel_sdk.robotis_def import DXL_HIBYTE
from dynamixel_sdk.robotis_def import DXL_HIWORD
from dynamixel_sdk.robotis_def import DXL_LOBYTE
from dynamixel_sdk.robotis_def import DXL_LOWORD
import numpy as np

from .constants import ADDR_GOAL_CURRENT
from .constants import ADDR_GOAL_POSITION
from .constants import ADDR_POSITION_D_GAIN
from .constants import ADDR_POSITION_I_GAIN
from .constants import ADDR_POSITION_P_GAIN
from .constants import ADDR_PRESENT_CURRENT
from .constants import ADDR_PRESENT_POS_VEL_CUR
from .constants import ADDR_PRESENT_POSITION
from .constants import ADDR_PRESENT_VELOCITY
from .constants import ADDR_TORQUE_ENABLE
from .constants import SIZE_GOAL_POSITION
from .constants import SIZE_PRESENT_CURRENT
from .constants import SIZE_PRESENT_POS_VEL_CUR
from .constants import SIZE_PRESENT_POSITION
from .constants import SIZE_PRESENT_VELOCITY

DEFAULT_POS_SCALE = 2.0 * np.pi / 4096  # 0.088 degrees per unit
DEFAULT_VEL_SCALE = 0.229 * 2.0 * np.pi / 60.0  # 0.229 rpm
DEFAULT_CUR_SCALE = 1.34

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamixelDriver:
    def __init__(
        self,
        servo_ids: Sequence[int],
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 4_000_000,
        pos_scale: float = DEFAULT_POS_SCALE,
        vel_scale: float = DEFAULT_VEL_SCALE,
        cur_scale: float = DEFAULT_CUR_SCALE,
        reading_interval: float = 0.0167,  # 60 Hz
        reading_retries: int = 5,
        writing_retries: int = 3,
        writing_retry_interval: float = 0.02,
    ):
        self.servo_ids = servo_ids
        self.port = port
        self.baud_rate = baud_rate
        self.pos_scale = pos_scale
        self.vel_scale = vel_scale
        self.cur_scale = cur_scale
        self.reading_interval = reading_interval
        self.reading_retries = reading_retries
        self.writing_retries = writing_retries
        self.writing_retry_interval = writing_retry_interval

        self._port_handler = PortHandler(port)
        self._packet_handler = Protocol2PacketHandler()

        self._joint_positions = np.zeros(len(servo_ids), dtype=np.int32)
        self._goal_positions = np.zeros(len(servo_ids), dtype=np.int32)
        self._joint_velocities = np.zeros(len(servo_ids), dtype=np.int32)
        self._joint_currents = np.zeros(len(servo_ids), dtype=np.int32)
        self._lock = Lock()

        # Reader for joint positions, velocities, and currents
        self._group_sync_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            ADDR_PRESENT_POS_VEL_CUR,
            SIZE_PRESENT_POS_VEL_CUR,
        )

        # Writer for goal positions
        self._group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            ADDR_GOAL_POSITION,
            SIZE_GOAL_POSITION,
        )

        # Open the port and set the baud rate
        if not self._port_handler.openPort():
            raise RuntimeError(f"Failed to open port {port}")

        if not self._port_handler.setBaudRate(baud_rate):
            raise RuntimeError(f"Failed to set baud rate {baud_rate}")

        for servo_id in servo_ids:
            if not self._group_sync_read.addParam(servo_id):
                raise RuntimeError(f"Failed to add servo ID {servo_id} to sync read group")

        # Reboot
        for servo_id in servo_ids:
            self._packet_handler.reboot(self._port_handler, servo_id)

        # Enable torque for each Dynamixel servo
        self._torque_enabled = False

        self._stop_thread = Event()
        self._start_reading_thread()

        atexit.register(self.close)

    def _reading_loop(self):
        retries = self.reading_retries
        while not self._stop_thread.is_set():
            time.sleep(self.reading_interval)
            with self._lock:
                dxl_comm_result = self._group_sync_read.fastSyncRead()
                if dxl_comm_result != COMM_SUCCESS:
                    retries -= 1
                    if retries <= 0:
                        logger.warning(
                            f"Failed to read data from Dynamixel servos after {self.reading_retries - retries} retries, data may be delayed or unavailable."
                        )
                    continue
                retries = self.reading_retries

                positions = np.zeros_like(self._joint_positions)
                goal_positions = np.zeros_like(self._goal_positions)
                velocities = np.zeros_like(self._joint_velocities)
                currents = np.zeros_like(self._joint_currents)

                for i, servo_id in enumerate(self.servo_ids):
                    if self._group_sync_read.isAvailable(servo_id, ADDR_PRESENT_POS_VEL_CUR, SIZE_PRESENT_POS_VEL_CUR):
                        positions[i] = np.int32(
                            np.uint32(
                                self._group_sync_read.getData(servo_id, ADDR_PRESENT_POSITION, SIZE_PRESENT_POSITION)
                            )
                        )
                        velocities[i] = np.int32(
                            np.uint32(
                                self._group_sync_read.getData(servo_id, ADDR_PRESENT_VELOCITY, SIZE_PRESENT_VELOCITY)
                            )
                        )
                        currents[i] = np.int32(
                            np.uint32(
                                self._group_sync_read.getData(servo_id, ADDR_PRESENT_CURRENT, SIZE_PRESENT_CURRENT)
                            )
                        )
                    if self._group_sync_read.isAvailable(servo_id, ADDR_GOAL_POSITION, SIZE_GOAL_POSITION):
                        goal_positions[i] = np.int32(
                            np.uint32(self._group_sync_read.getData(servo_id, ADDR_GOAL_POSITION, SIZE_GOAL_POSITION))
                        )

                self._joint_positions = positions
                self._goal_positions = goal_positions
                self._joint_velocities = velocities
                self._joint_currents = currents

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self._reading_loop, daemon=True)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def _write_with_retry(self, write_func, *args, err_msg: str = "", **kwargs):
        attempt = 0
        while attempt <= self.writing_retries:
            result = write_func(*args, **kwargs)
            # write1ByteTxRx, write4ByteTxRx等返回 (comm_result, dxl_error)
            if isinstance(result, tuple) and len(result) == 2:
                comm_result, dxl_error = result
            else:
                comm_result, dxl_error = result, 0
            if comm_result == COMM_SUCCESS and dxl_error == 0:
                return True
            logger.warning(
                f"Attempt {attempt + 1}/{self.writing_retries + 1}: {err_msg} (COMM_RESULT={comm_result}, ERROR={dxl_error})"
            )
            attempt += 1
            time.sleep(self.writing_retry_interval)
        raise RuntimeError(f"{err_msg} after {self.writing_retries + 1} attempts")

    def _set_servo_torque(self, servo_id: int, *, enable: bool):
        self._write_with_retry(
            self._packet_handler.write1ByteTxRx,
            self._port_handler,
            servo_id,
            ADDR_TORQUE_ENABLE,
            int(enable),
            err_msg=f"Failed to set torque mode for Dynamixel ID {servo_id}",
        )

    def handle_packet_result(
        self,
        comm_result: int,
        dxl_error: int | None = None,
        dxl_id: int | None = None,
        context: str | None = None,
    ):
        """Handles the result from a communication request."""
        error_message = None
        if comm_result != COMM_SUCCESS:
            error_message = self._packet_handler.getTxRxResult(comm_result)
        elif dxl_error is not None:
            error_message = self._packet_handler.getRxPacketError(dxl_error)
        if error_message:
            if dxl_id is not None:
                error_message = f"[Motor ID: {dxl_id}] {error_message}"
            if context is not None:
                error_message = f"> {context}: {error_message}"
            logger.error(error_message)
            return False
        return True

    def close(self):
        self._stop_thread.set()
        self._reading_thread.join()
        self.torque_enabled = False
        self._port_handler.closePort()

    def set_position_p_gains(self, gains: list[float]):
        with self._lock:
            for dxl_id, gain in zip(self.servo_ids, gains, strict=False):
                self._write_with_retry(
                    self._packet_handler.write2ByteTxRx,
                    self._port_handler,
                    dxl_id,
                    ADDR_POSITION_P_GAIN,
                    int(gain),
                    err_msg=f"Failed to set P gain for Dynamixel ID {dxl_id}",
                )

    def set_position_i_gains(self, gains: list[float]):
        with self._lock:
            for dxl_id, gain in zip(self.servo_ids, gains, strict=False):
                self._write_with_retry(
                    self._packet_handler.write2ByteTxRx,
                    self._port_handler,
                    dxl_id,
                    ADDR_POSITION_I_GAIN,
                    int(gain),
                    err_msg=f"Failed to set I gain for Dynamixel ID {dxl_id}",
                )

    def set_position_d_gains(self, gains: list[float]):
        with self._lock:
            for dxl_id, gain in zip(self.servo_ids, gains, strict=False):
                self._write_with_retry(
                    self._packet_handler.write2ByteTxRx,
                    self._port_handler,
                    dxl_id,
                    ADDR_POSITION_D_GAIN,
                    int(gain),
                    err_msg=f"Failed to set D gain for Dynamixel ID {dxl_id}",
                )

    def set_goal_currents(self, currents: list[float]):
        with self._lock:
            for dxl_id, current in zip(self.servo_ids, currents, strict=False):
                self._write_with_retry(
                    self._packet_handler.write2ByteTxRx,
                    self._port_handler,
                    dxl_id,
                    ADDR_GOAL_CURRENT,
                    int(current),
                    err_msg=f"Failed to set goal current for Dynamixel ID {dxl_id}",
                )

    @property
    def torque_enabled(self) -> bool:
        return self._torque_enabled

    @torque_enabled.setter
    def torque_enabled(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        with self._lock:
            for dxl_id in self.servo_ids:
                self._set_servo_torque(dxl_id, enable=enable)

        self._torque_enabled = enable

    @property
    def joint_positions(self) -> np.ndarray:
        return self._joint_positions.copy() * self.pos_scale

    @property
    def goal_positions(self) -> np.ndarray:
        return self._goal_positions.copy() * self.pos_scale

    @goal_positions.setter
    def goal_positions(self, positions: Sequence[float]):
        self._group_sync_write.clearParam()
        assert len(positions) == len(self.servo_ids), "Length of goal_positions must match number of servos"
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set goal positions")

        error_ids = []
        for servo_id, position in zip(self.servo_ids, positions, strict=True):
            position_value = int(position / self.pos_scale)
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position_value)),
                DXL_HIBYTE(DXL_LOWORD(position_value)),
                DXL_LOBYTE(DXL_HIWORD(position_value)),
                DXL_HIBYTE(DXL_HIWORD(position_value)),
            ]

            add_param_result = self._group_sync_write.addParam(servo_id, param_goal_position)
            if not add_param_result:
                error_ids.append(servo_id)

        if error_ids:
            logger.error(f"Failed to set joint positions for Dynamixel IDs: {error_ids}")

        with self._lock:
            comm_result = self._group_sync_write.txPacket()
            if comm_result != COMM_SUCCESS:
                self.handle_packet_result(comm_result, context="sync_write")

        self._group_sync_write.clearParam()

    @property
    def joint_velocities(self) -> np.ndarray:
        return self._joint_velocities.copy() * self.vel_scale

    @property
    def joint_currents(self) -> np.ndarray:
        return self._joint_currents.copy() * self.cur_scale


if __name__ == "__main__":
    driver = DynamixelDriver(servo_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], port="/dev/ttyUSB0")
    driver.set_position_p_gains([600, 400, 400, 400, 600, 400, 400, 400, 600, 400, 400, 400, 600, 400, 400, 400])
    driver.set_position_i_gains([0] * 16)
    driver.set_position_d_gains([150, 200, 200, 200, 150, 200, 200, 200, 150, 200, 200, 200, 150, 200, 200, 200])
    driver.set_goal_currents([500] * 16)
    driver.torque_enabled = True

    driver.goal_positions = np.ones(16) * np.pi
    print("Positions:", driver.joint_positions)
    time.sleep(10)

    del driver
