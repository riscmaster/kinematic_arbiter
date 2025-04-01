"""Setup configuration for the kinematic_arbiter package."""

from setuptools import setup, find_packages
from glob import glob

package_name = "kinematic_arbiter"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            ["launch/simplified_demo.launch.py"],
        ),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Spencer Maughan",
    author_email="spence.maughan@gmail.com",
    maintainer="Spencer Maughan",
    maintainer_email="spence.maughan@gmail.com",
    keywords=["ROS"],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Software Development",
    ],
    description="Single DOF demo package",
    license="Apache License, Version 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "kalman_filter_node = single_dof_demo.ros2.nodes.kalman_filter_node:main",
            "mediated_filter_node = single_dof_demo.ros2.nodes.mediated_filter_node:main",
            "signal_generator_node = single_dof_demo.ros2.nodes.signal_generator_node:main",
        ],
    },
)
