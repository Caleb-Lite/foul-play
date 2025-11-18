import os
from .team_converter import export_to_packed, export_to_dict
from .meta_selector import MetaTeamSelector

TEAM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teams")


def load_team(name, pokemon_format=None, meta_selector=None):
    if name is None:
        return "null", "", ""

    selector = meta_selector or MetaTeamSelector()
    file_path = selector.resolve_team_path(name, pokemon_format)

    with open(file_path, "r") as f:
        team_export = f.read()

    relative_name = os.path.relpath(file_path, TEAM_DIR)

    return (
        export_to_packed(team_export),
        export_to_dict(team_export),
        relative_name,
    )
