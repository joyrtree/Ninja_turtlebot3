from glob import glob
from setuptools import find_packages, setup

setup(
    name="lane_detection_fusion",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/lane_detection_fusion"]),
        ("share/lane_detection_fusion", ["package.xml", "README.md"]),
        ("share/lane_detection_fusion/config", glob("config/*.json")),
        ("share/lane_detection_fusion/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="TurtleBot AutoRace team",
    maintainer_email="joyrtree@ajou.ac.kr",
    description="Fresh-frame lane geometry, boundary identity and explicit loss handling",
    license="Proprietary",
    entry_points={"console_scripts": ["fusion = lane_detection_fusion.node:main"]},
)
