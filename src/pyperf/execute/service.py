import rpyc
import time
from threading import Thread, Event
from rpyc.utils.server import ThreadPoolServer
from r2e_test_server.server import R2EService


server_stop_event = Event()


def start_server_nonblocking(port: int):
    server = ThreadPoolServer(R2EService(), port=port)

    def run_server():
        server.start()
        server_stop_event.wait()
        server.close()
        print(f"Server on port {port} stopped")

    server_thread = Thread(target=run_server)
    server_thread.start()
    return server_thread


def stop_server():
    server_stop_event.set()


class ServiceManager:

    running_ports = set()
    server_threads = {}

    @staticmethod
    def get_service(repo_id: str, port: int, local: bool = False):

        if local:
            # start a new server at given port
            if not port in ServiceManager.running_ports:
                t = start_server_nonblocking(port)
                ServiceManager.running_ports.add(port)
                ServiceManager.server_threads[port] = t
                time.sleep(3)

            # connect to the server
            _, conn = ServiceManager.connect_local(port)

            return None, conn

        return ServiceManager.get_service_docker(repo_id, port)

    @staticmethod
    def get_service_docker(repo_id: str, port: int):
        raise NotImplementedError

    @staticmethod
    def connect_local(port: int):
        try:
            conn = rpyc.connect("localhost", port)
        except Exception as e:
            print(f"Connection error -- {repr(e)} -- {port}")
            raise e
        return None, conn

    @staticmethod
    def shutdown():
        for port in list(ServiceManager.running_ports):
            print(f"Closing connection on port {port}")
            try:
                conn = rpyc.connect("localhost", port)
                service = conn.root
                service.stop_server()
                conn.close()
            except Exception as e:
                print(f"Error closing connection on port {port}: {e}")

        # Stop all server threads
        stop_server()
        for port, thread in ServiceManager.server_threads.items():
            thread.join(timeout=5)
            if thread.is_alive():
                ServiceManager.force_stop_thread(thread)

        ServiceManager.running_ports.clear()
        ServiceManager.server_threads.clear()
        print("All connections closed and servers stopped")

    @staticmethod
    def close_connection(port):
        try:
            conn = rpyc.connect("localhost", port)
            service = conn.root
            service.stop_server()
            conn.close()
        except Exception as e:
            print(f"Error closing connection on port {port}: {e}")

        if port in ServiceManager.running_ports:
            ServiceManager.running_ports.remove(port)

        if port in ServiceManager.server_threads:
            thread = ServiceManager.server_threads[port]
            thread.join(timeout=5)

            if thread.is_alive():
                ServiceManager.force_stop_thread(thread)

            del ServiceManager.server_threads[port]

    @staticmethod
    def force_stop_thread(thread):
        if hasattr(thread, "_tstate_lock"):
            if thread._tstate_lock.locked():
                thread._tstate_lock.release()
        thread._stop()
