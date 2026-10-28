#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time")

    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="true")

    rviz_goal_relay = Node(
        package="asl_tb3_lib",
        executable="rviz_goal_relay.py",
        parameters=[{"output_channel": "/cmd_nav"}],
    )

    # state publisher for turtlebot
    state_publisher = Node(
        executable="state_publisher.py",
        package="asl_tb3_lib",
    )

    navigator = Node (
        package="autonomy_repo",
        executable="navigator.py",
        parameters=[{"Use_sim_time": use_sim_time}],
        output='screen',
        emulate_tty=True
    )


    rviz_launch = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
        ),
        launch_arguments={
            "config": PathJoinSubstitution(
                [
                    FindPackageShare("autonomy_repo"),
                    "rviz",
                    "default.rviz",
                ]
            ),
            "use_sim_time": use_sim_time,
        }.items(),
    )

    return LaunchDescription([
        use_sim_time_arg,
        rviz_goal_relay,
        state_publisher,
        navigator,
        rviz_launch
    ])