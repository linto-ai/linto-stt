import yaml
from flask_swagger_ui import get_swaggerui_blueprint


def setupSwaggerUI(app, args):
    """Setup Swagger UI within the app"""
    with open(args.swagger_path, "r") as yml_file:
        swagger_yml = yaml.load(yml_file, Loader=yaml.Loader)
    swaggerui = get_swaggerui_blueprint(
        # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
        args.swagger_prefix + args.swagger_url,
        args.swagger_path,
        config={  # Swagger UI config overrides
            "app_name": "LinTO Platform STT",
            "spec": swagger_yml,
        },
    )
    app.register_blueprint(swaggerui, url_prefix=args.swagger_url)
