"""Entry point: python run.py [--host H] [--port P] [--config config.yaml]"""
import argparse

import uvicorn

from server.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Semantic observability server")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    host = args.host or config.get("api.host", "0.0.0.0")
    port = args.port or int(config.get("api.port", 8000))

    import os
    os.environ["APP_CONFIG_PATH"] = args.config
    uvicorn.run("server.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
