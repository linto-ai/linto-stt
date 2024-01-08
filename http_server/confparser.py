import argparse
import os

__all__ = ["createParser"]


def createParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # GUNICORN
    parser.add_argument("--service_port", type=int, help="Service port", default=80)
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of Gunicorn workers (default=CONCURRENCY + 1)",
        default=int(os.environ.get("CONCURRENCY", 1)) + 1,
    )

    # SWAGGER
    parser.add_argument("--swagger_url", type=str, help="Swagger interface url", default="/docs")
    parser.add_argument(
        "--swagger_prefix",
        type=str,
        help="Swagger prefix",
        default=os.environ.get("SWAGGER_PREFIX", ""),
    )
    parser.add_argument(
        "--swagger_path",
        type=str,
        help="Swagger file path",
        default=os.environ.get("SWAGGER_PATH", "/usr/src/app/document/swagger.yml"),
    )

    # MISC
    parser.add_argument("--debug", action="store_true", help="Display debug logs")

    return parser
