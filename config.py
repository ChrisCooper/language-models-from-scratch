import os
from dotenv import dotenv_values, find_dotenv
from pydantic import BaseModel

ENV_NAME = os.environ.get("ENV_NAME", "dev")


class MyConfig(BaseModel):
    ENV_NAME: str = "dev"
    APP_NAME: str


_CONFIG: MyConfig | None = None


def get_config():
    global _CONFIG
    if _CONFIG is None:
        reload_config()
    return _CONFIG


def reload_config():
    global _CONFIG
    print(f"Loading config for ENV_NAME={ENV_NAME}")
    _CONFIG = MyConfig.validate(load_config(ENV_NAME))
    print(f"Reloaded config")


def load_config(env_name: str):
    conf = {
        "ENV_NAME": env_name,
        **dotenv_values(find_dotenv(".env.shared_secrets")),  # load shared secrets
        **dotenv_values(find_dotenv(".env.shared")),  # load shared general variables
        **dotenv_values(find_dotenv(f".env.{env_name}")),  # override with env-specific variables
        **os.environ,  # override with environment variables
    }
    return conf


if ENV_NAME != "testing":
    get_config()
