import socket
import numpy as np
import struct
import yaml
import time

def get_packet(tcp_socket, packet_size):
    data = b''
    while len(data) < packet_size:
        try:
            packet = tcp_socket.recv(packet_size - len(data))
        except socket.timeout:
            raise TimeoutError(
                f"Timeout waiting for packet: got {len(data)}/{packet_size} bytes"
            )

        if not packet:
            raise ConnectionError(
                f"Socket closed while waiting for packet: got {len(data)}/{packet_size} bytes"
            )

        data += packet

    return data


def send_packet(tcp_socket, packet_format, data):
    packed_data = struct.pack(packet_format, *data)
    tcp_socket.sendall(packed_data)


class NetworkAdaptor:
    INITIAL_PACKET_FORMAT = "<26i296x"
    GETTING_PACKET_FORMAT = "=27d"
    SENDING_PACKET_FORMAT = "<5d"
    INITIAL_PACKET_SIZE = 400
    GETTING_PACKET_SIZE = 216
    SENDING_PACKET_SIZE = 40

    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(10.0)
        self.host = self.config['host']
        self.port = self.config['port']

    def load_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config

    def connect(self):
        print(f"[NetworkAdaptor] connecting to {self.host}:{self.port}", flush=True)
        self.socket.connect((self.host, self.port))
        print(f"[NetworkAdaptor] connected to {self.host}:{self.port}", flush=True)

    def reconnect(self, retries=5, delay=0.2):
        try:
            self.socket.close()
        except Exception:
            pass

        last_err = None

        for i in range(retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10.0)
                print(
                    f"[NetworkAdaptor] reconnecting to {self.host}:{self.port}, "
                    f"try {i+1}/{retries}",
                    flush=True
                )
                self.socket.connect((self.host, self.port))
                print(f"[NetworkAdaptor] reconnected to {self.host}:{self.port}", flush=True)
                return
            except Exception as e:
                last_err = e
                try:
                    self.socket.close()
                except Exception:
                    pass
                time.sleep(delay)

        raise ConnectionError(f"Failed to reconnect to {self.host}:{self.port}: {last_err}")

    def send_initial_packet(self, initial_data):
        send_packet(self.socket, self.INITIAL_PACKET_FORMAT, initial_data)

    def get_observation_packet(self):
        data = get_packet(self.socket, self.GETTING_PACKET_SIZE)
        unpacked_data = np.array(
            struct.unpack(self.GETTING_PACKET_FORMAT, data),
            dtype=np.float64
        )
        return unpacked_data

    def send_action_packet(self, action):
        send_packet(self.socket, self.SENDING_PACKET_FORMAT, action)