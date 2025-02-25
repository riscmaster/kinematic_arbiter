"""Setup configuration for the kinematic_arbiter package."""

from setuptools import find_namespace_packages, setup

package_name = "kinematic_arbiter"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_namespace_packages(where="src"),
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
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Spencer Maughan",
    maintainer_email="your.email@example.com",
    description=(
        "Mediated Kalman filter implementation \
                 for robust state estimation"
    ),
    license="MIT",
    tests_require=["pytest"],
)
