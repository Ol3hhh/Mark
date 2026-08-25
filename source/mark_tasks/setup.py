"""Installation script for the mark_tasks package"""

from setuptools import setup

setup(
    name="mark_tasks",
    version="0.1.0",
    description="Isaac Lab RL tasks for the Mark robot series",
    packages=[
        "mark_tasks",
        "mark_tasks.robots",
        "mark_tasks.tasks",
        "mark_tasks.tasks.locomotion",
        "mark_tasks.tasks.locomotion.mark_v1",
        "mark_tasks.tasks.locomotion.mark_v1.agents",
        "mark_tasks.tasks.locomotion.mark_v1.mdp",
    ],
    package_dir={"mark_tasks": "."},
    python_requires=">=3.10",
    zip_safe=False,
)
