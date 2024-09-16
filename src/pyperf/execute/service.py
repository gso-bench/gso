import rpyc


class ServiceManager:
    @staticmethod
    def get_service(repo_id: str, port: int, local: bool = False):
        if local:
            return ServiceManager.get_service_local(port)
        return ServiceManager.get_service_docker(repo_id, port)

    @staticmethod
    def get_service_local(port: int):
        try:
            conn = rpyc.connect("localhost", port)
        except Exception as e:
            print(f"Connection error -- {repr(e)}")
            raise e
        return None, conn

    @staticmethod
    def get_service_docker(repo_id: str, port: int):
        # simulator = DockerSimulator(repo_id=repo_id, port=port)
        # try:
        #     conn = rpyc.connect(
        #         "localhost",
        #         port,
        #         keepalive=True,
        #         config={"sync_request_timeout": 180},
        #     )
        # except Exception as e:
        #     print(f"Connection error -- {repo_id} -- {repr(e)}")
        #     simulator.stop_container()
        #     raise e
        # return simulator, conn
        raise NotImplementedError
