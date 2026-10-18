from glob import glob

from setuptools import setup

package_name = "openarm_camera"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="USB (V4L2) camera publisher for the OpenArm head camera",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_publisher = openarm_camera.camera_publisher:main",
        ],
    },
)
