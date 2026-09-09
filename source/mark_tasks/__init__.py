"""Import packeges into Isaac Lab"""

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp", "agents"]
import_packages("mark_tasks.tasks", _BLACKLIST_PKGS)
