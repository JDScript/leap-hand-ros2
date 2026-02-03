from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                name="leap",
                package="leap_driver",
                executable="leap.py",
                output="screen",
            )
        ]
    )
