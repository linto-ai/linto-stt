import argparse
import os

__all__ = ["createParser"]


def createParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # SERVICE
    parser.add_argument(
        "--service_name",
        type=str,
        help="Service Name",
        default=os.environ.get("SERVICE_NAME", "stt"),
    )

    # MODELS
    parser.add_argument("--am_path", type=str, help="Acoustic Model Path", default="/opt/models/AM")
    parser.add_argument("--lm_path", type=str, help="Decoding graph path", default="/opt/models/LM")
    parser.add_argument(
        "--config_path", type=str, help="Configuration files path", default="/opt/config"
    )

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
