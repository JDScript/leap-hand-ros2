#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import JointState

from leap_driver.dynamixel.driver import DynamixelDriver


class Leap(Node):
    def __init__(
        self,
        node_name="Leap",
        *,
        context=None,
        cli_args=None,
        namespace="leap",
        use_global_arguments=True,
        enable_rosout=True,
        start_parameter_services=True,
        parameter_overrides=None,
        allow_undeclared_parameters=False,
        automatically_declare_parameters_from_overrides=False,
        enable_logger_service=False,
    ):
        super().__init__(
            node_name,
            context=context,
            cli_args=cli_args,
            namespace=namespace,
            use_global_arguments=use_global_arguments,
            enable_rosout=enable_rosout,
            start_parameter_services=start_parameter_services,
            parameter_overrides=parameter_overrides,
            allow_undeclared_parameters=allow_undeclared_parameters,
            automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides,
            enable_logger_service=enable_logger_service,
        )

        self.port = self.declare_parameter("port", "/dev/ttyUSB0").get_parameter_value().string_value
        self.baud_rate = self.declare_parameter("baudrate", 4_000_000).get_parameter_value().integer_value
        self.kP = self.declare_parameter("kP", 600.0).get_parameter_value().double_value
        self.kI = self.declare_parameter("kI", 0.0).get_parameter_value().double_value
        self.kD = self.declare_parameter("kD", 200.0).get_parameter_value().double_value
        self.goal_current = self.declare_parameter("goal_current", 500.0).get_parameter_value().double_value

        self.qos_profile = QoSProfile(depth=10)
        self.qos_profile.reliability = QoSReliabilityPolicy.RELIABLE
        self.qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.qos_profile.history = QoSHistoryPolicy.KEEP_LAST

        # Initialize Driver
        self.driver = DynamixelDriver(
            servo_ids=list(range(16)),
            port=self.port,
            baud_rate=self.baud_rate,
        )
        self.driver.set_position_p_gains([self.kP * 0.75, self.kP, self.kP, self.kP] * 4)
        self.driver.set_position_i_gains([self.kI * 0.75, self.kI, self.kI, self.kI] * 4)
        self.driver.set_position_d_gains([self.kD * 0.75, self.kD, self.kD, self.kD] * 4)
        self.driver.set_goal_currents([self.goal_current] * 16)
        self.driver.torque_enabled = True

        self.driver.goal_positions = np.zeros(16) + np.pi

        # Publishers
        self.publisher = self.create_publisher(
            JointState,
            "joint_states",
            self.qos_profile,
        )

        # Subscribers
        self.create_subscription(
            JointState,
            "goal_positions",
            self.goal_positions_callback,
            self.qos_profile,
        )

        # Timers
        self.timer = self.create_timer(0.0167, self.timer_callback)

    def goal_positions_callback(self, msg: JointState):
        self.driver.goal_positions = np.array(msg.position) + np.pi

    def timer_callback(self):
        # Publish joint positions
        state = JointState()
        state.position = (self.driver.joint_positions - np.pi).tolist()
        state.velocity = self.driver.joint_velocities.tolist()
        state.effort = self.driver.joint_currents.tolist()
        self.publisher.publish(state)


def main(args=None):
    rclpy.init(args=args)
    leap = Leap()
    try:
        rclpy.spin(leap)
    except KeyboardInterrupt:
        pass
    finally:
        leap.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
