from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("drive_enabled", default_value="false"),
        DeclareLaunchArgument("camera_topic", default_value="/camera_raw_16_9"),
        Node(
            package="lane_detection_fusion", executable="fusion", output="screen",
            parameters=[{
                "drive_enabled": ParameterValue(LaunchConfiguration("drive_enabled"), value_type=bool),
                "camera_topic": LaunchConfiguration("camera_topic"),
            }],
        ),
    ])
