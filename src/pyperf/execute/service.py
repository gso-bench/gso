import rpyc


class ServiceManager:
    def __init__(self, repo_id: str, port: int, local: bool = False):
        self.repo_id = repo_id
        self.port = port
        self.local = local

    def get_service(self):
        if self.local:
            return self.get_service_local()
        return self.get_service_docker()

    def get_service_local(self):
        try:
            conn = rpyc.connect("localhost", self.port)
        except Exception as e:
            print(f"Connection error -- {repr(e)}")
            raise e
        return conn

    def get_service_docker(self):
        # simulator = DockerSimulator(repo_id=self.repo_id, port=self.port)
        # try:
        #     conn = rpyc.connect(
        #         "localhost",
        #         self.port,
        #         keepalive=True,
        #         config={"sync_request_timeout": 180},
        #     )
        # except Exception as e:
        #     print(f"Connection error -- {self.repo_id} -- {repr(e)}")
        #     simulator.stop_container()
        #     raise e
        # return simulator, conn
        raise NotImplementedError
