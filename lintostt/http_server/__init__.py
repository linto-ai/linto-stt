import logging
import json

from fastapi import FastAPI, APIRouter, Request, UploadFile


logger = logging.getLogger("__stt-standalone-worker__")
logger.setLevel(logging.INFO)

class Http_Server:
    def __init__(self, name: str, model, use_gpu, decode, load_wave_buffer, warmup):
        self.name = name
        self.model = model
        self.use_gpu = use_gpu
        self.decode = decode
        self.load_wave_buffer = load_wave_buffer
        self.warmup = warmup


        self.router = APIRouter()
        self.router.add_api_route("/hello", self.hello, methods=["GET"])
        self.router.add_api_route("/healthcheck", self.healthcheck, methods=["GET"])
        self.router.add_api_route("/transcribe", self.transcribe, methods=["POST"])

    def hello(self):
        return {"Hello": self.name}
    
    def healthcheck(self):
        return {"status": "ok"}
    
    def transcribe(self, request: Request, file: UploadFile):
        try:
            logger.info("Transcribe request received")
            if request.headers.get("accept").lower() == "application/json":
                join_metadata = True
            elif request.headers.get("accept").lower() == "text/plain":
                join_metadata = False
            else:
                raise ValueError(
                    f"Not accepted header (accept={request.headers.get('accept')} should be either application/json or text/plain)"
                )
            
            if not file.filename:
                raise ValueError(f"No audio file was uploaded (missing 'file' key)")
            
            file_buffer = file.file.read()
            language = request.query_params.get("language")
            logger.info(f"Transcribing {file.filename} in {language}")

            audio_data = self.load_wave_buffer(file_buffer)
            transcription= self.decode(audio_data, self.model, join_metadata, language=language)

            if join_metadata:
                return json.dumps(transcription, ensure_ascii=False), 200
            else:
                return transcription, 200

        except Exception as error:
            import traceback
            logger.error(traceback.format_exc())
            logger.error(repr(error))
            return "Server Error: {}".format(str(error)), (
                400 if isinstance(error, ValueError) else 500
            )

def main(model, use_gpu, decode, load_wave_buffer, warmup):
    app = FastAPI()
    server = Http_Server("LinTO-STT", model, use_gpu, decode, load_wave_buffer, warmup)
    app.include_router(server.router)
    return app
