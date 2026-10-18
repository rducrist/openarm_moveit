from glob import glob

from setuptools import setup

package_name = "openarm_ee_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/urdf", glob("urdf/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="End-effector pose control for OpenArm with MoveIt 2",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ee_pose_server = openarm_ee_control.ee_pose_server:main",
            "send_pose = openarm_ee_control.send_pose:main",
            "gravity_comp = openarm_ee_control.gravity_comp:main",
            "traj_replay = openarm_ee_control.traj_replay:main",
            "table_collision = openarm_ee_control.table_collision:main",
        ],
    },
)
