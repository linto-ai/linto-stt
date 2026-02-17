import os
from urllib.parse import urlparse, urlunparse

from celery import Celery

celery = Celery(__name__)
service_name = os.environ.get("SERVICE_NAME", "stt")
broker_url = os.environ.get("SERVICES_BROKER", "redis://localhost:6379")
if os.environ.get("BROKER_PASS", False):
    components = broker_url.split("//")
    broker_url = f'{components[0]}//:{os.environ.get("BROKER_PASS")}@{components[1]}'

parsed = urlparse(broker_url)
base_url = urlunparse(parsed._replace(path=""))
celery.conf.broker_url = f"{base_url}/0"
celery.conf.result_backend = f"{base_url}/1"
celery.conf.task_acks_late = False
celery.conf.task_track_started = True
celery.conf.broker_transport_options = {"visibility_timeout": float("inf")}
# celery.conf.result_backend_transport_options = {"visibility_timeout": float("inf")}
# celery.conf.result_expires = 3600 * 24

# Queues
celery.conf.update(
    {
        "task_routes": {
            "transcribe_task": {"queue": service_name},
        }
    }
)

# logger.info(
#     f"Celery configured for broker located at {broker_url} with service name {service_name}"
# )
