from setuptools import setup

package_name = "openarm_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Step-wise deployment of R2R2R diffusion policies on OpenArm",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy_node = openarm_policy.policy_node:main",
        ],
    },
)
