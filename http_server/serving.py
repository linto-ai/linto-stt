import gunicorn.app.base
import gevent.pywsgi
import gevent.monkey
gevent.monkey.patch_all()


class GunicornServing(gunicorn.app.base.BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

class GeventServing():

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app

    def run(self):
        bind = self.options.get('bind', "0.0.0.0:8080")
        workers = self.options.get('workers', 1)
        listener = bind.split(':')
        try:
            assert len(listener) == 2
            listener = (listener[0], int(listener[1]))
        except:
            print(f"Invalid bind address {bind}")

        server = gevent.pywsgi.WSGIServer(listener, self.application, spawn = workers)
        server.serve_forever()

