import os

from celery import Celery

celery = Celery(__name__)
service_name = os.environ.get("SERVICE_NAME", "stt")
broker_url = os.environ.get("SERVICES_BROKER", "redis://localhost:6379")
if os.environ.get("BROKER_PASS", False):
    components = broker_url.split("//")
    broker_url = f'{components[0]}//:{os.environ.get("BROKER_PASS")}@{components[1]}'

celery.conf.broker_url = f"{broker_url}/0"
celery.conf.result_backend = f"{broker_url}/1"
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
