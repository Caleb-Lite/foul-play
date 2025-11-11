import argparse
import logging
import os
import sys
from enum import Enum, auto
from logging.handlers import RotatingFileHandler
from typing import Optional

from dotenv import load_dotenv


class CustomFormatter(logging.Formatter):
    def format(self, record):
        lvl = "{}".format(record.levelname)
        return "{} {}".format(lvl.ljust(8), record.msg)


class CustomRotatingFileHandler(RotatingFileHandler):
    def __init__(self, file_name, **kwargs):
        self.base_dir = "logs"
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        super().__init__("{}/{}".format(self.base_dir, file_name), **kwargs)

    def do_rollover(self, new_file_name):
        new_file_name = new_file_name.replace("/", "_")
        self.baseFilename = "{}/{}".format(self.base_dir, new_file_name)
        self.doRollover()


def init_logging(level, log_to_file):
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    requests_logger = logging.getLogger("urllib3")
    requests_logger.setLevel(logging.INFO)

    # Gets the root logger to set handlers/formatters
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(CustomFormatter())
    logger.addHandler(stdout_handler)
    FoulPlayConfig.stdout_log_handler = stdout_handler

    if log_to_file:
        file_handler = CustomRotatingFileHandler("init.log")
        file_handler.setLevel(logging.DEBUG)  # file logs are always debug
        file_handler.setFormatter(CustomFormatter())
        logger.addHandler(file_handler)
        FoulPlayConfig.file_log_handler = file_handler


class SaveReplay(Enum):
    always = auto()
    never = auto()
    on_loss = auto()


class BotModes(Enum):
    challenge_user = auto()
    accept_challenge = auto()
    search_ladder = auto()


class _FoulPlayConfig:
    websocket_uri: str
    username: str
    password: str
    user_id: str
    avatar: str
    bot_mode: BotModes
    pokemon_format: str = ""
    smogon_stats: str = None
    search_time_ms: int
    parallelism: int
    run_count: int
    team_name: str
    user_to_challenge: str
    save_replay: SaveReplay
    room_name: str
    log_level: str
    log_to_file: bool
    manual_mode: bool = False
    stdout_log_handler: logging.StreamHandler
    file_log_handler: Optional[CustomRotatingFileHandler]

    def configure(self):
        # Load environment variables from .env file
        load_dotenv()

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--websocket-uri",
            default=os.getenv("WEBSOCKET_URI"),
            help="The PokemonShowdown websocket URI, e.g. wss://sim3.psim.us/showdown/websocket",
        )
        parser.add_argument("--ps-username", default=os.getenv("PS_USERNAME"))
        parser.add_argument("--ps-password", default=os.getenv("PS_PASSWORD"))
        parser.add_argument("--ps-avatar", default=os.getenv("PS_AVATAR"))
        parser.add_argument(
            "--bot-mode", default=os.getenv("BOT_MODE"), choices=[e.name for e in BotModes]
        )
        parser.add_argument(
            "--user-to-challenge",
            default=os.getenv("USER_TO_CHALLENGE"),
            help="If bot_mode is `challenge_user`, this is required",
        )
        parser.add_argument(
            "--pokemon-format", default=os.getenv("POKEMON_FORMAT"), help="e.g. gen9randombattle"
        )
        parser.add_argument(
            "--smogon-stats-format",
            default=os.getenv("SMOGON_STATS_FORMAT"),
            help="Overwrite which smogon stats are used to infer unknowns. If not set, defaults to the --pokemon-format value.",
        )
        parser.add_argument(
            "--search-time-ms",
            type=int,
            default=int(os.getenv("SEARCH_TIME_MS", "100")),
            help="Time to search per battle in milliseconds",
        )
        parser.add_argument(
            "--search-parallelism",
            type=int,
            default=int(os.getenv("SEARCH_PARALLELISM", "1")),
            help="Number of states to search in parallel",
        )
        parser.add_argument(
            "--run-count",
            type=int,
            default=int(os.getenv("RUN_COUNT", "1")),
            help="Number of PokemonShowdown battles to run",
        )
        parser.add_argument(
            "--team-name",
            default=os.getenv("TEAM_NAME"),
            help="Which team to use. Can be a filename or a foldername relative to ./teams/teams/. "
            "If a foldername, a random team from that folder will be chosen each battle. "
            "If not set, defaults to the --pokemon-format value.",
        )
        parser.add_argument(
            "--save-replay",
            default=os.getenv("SAVE_REPLAY", "never"),
            choices=[e.name for e in SaveReplay],
            help="When to save replays",
        )
        parser.add_argument(
            "--room-name",
            default=os.getenv("ROOM_NAME"),
            help="If bot_mode is `accept_challenge`, the room to join while waiting",
        )
        parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "DEBUG"), help="Python logging level")
        parser.add_argument(
            "--log-to-file",
            action="store_true",
            default=os.getenv("LOG_TO_FILE", "").lower() in ("true", "1", "yes"),
            help="When enabled, DEBUG logs will be written to a file in the logs/ directory",
        )
        parser.add_argument(
            "--manual-mode",
            action="store_true",
            default=os.getenv("MANUAL_MODE", "").lower() in ("true", "1", "yes"),
            help="When enabled, the bot will suggest moves but not execute them. You make moves in Pokemon Showdown directly.",
        )

        args = parser.parse_args()

        # Validate required parameters
        if not args.websocket_uri:
            parser.error("--websocket-uri is required (or set WEBSOCKET_URI environment variable)")
        if not args.ps_username:
            parser.error("--ps-username is required (or set PS_USERNAME environment variable)")
        if not args.ps_password:
            parser.error("--ps-password is required (or set PS_PASSWORD environment variable)")
        if not args.bot_mode:
            parser.error("--bot-mode is required (or set BOT_MODE environment variable)")
        if not args.pokemon_format:
            parser.error("--pokemon-format is required (or set POKEMON_FORMAT environment variable)")

        self.websocket_uri = args.websocket_uri
        self.username = args.ps_username
        self.password = args.ps_password
        self.avatar = args.ps_avatar
        self.bot_mode = BotModes[args.bot_mode]
        self.pokemon_format = args.pokemon_format
        self.smogon_stats = args.smogon_stats_format
        self.search_time_ms = args.search_time_ms
        self.parallelism = args.search_parallelism
        self.run_count = args.run_count
        self.team_name = args.team_name or self.pokemon_format
        self.user_to_challenge = args.user_to_challenge
        self.save_replay = SaveReplay[args.save_replay]
        self.room_name = args.room_name
        self.log_level = args.log_level
        self.log_to_file = args.log_to_file
        self.manual_mode = args.manual_mode

        self.validate_config()

    def requires_team(self) -> bool:
        return not (
            "random" in self.pokemon_format or "battlefactory" in self.pokemon_format
        )

    def validate_config(self):
        if self.bot_mode == BotModes.challenge_user:
            assert (
                self.user_to_challenge is not None
            ), "If bot_mode is `CHALLENGE_USER`, you must declare USER_TO_CHALLENGE"


FoulPlayConfig = _FoulPlayConfig()
