import json
import logging
import socket
from json.decoder import JSONDecodeError
from time import sleep


class ChunkedSocketCommunicator(object):
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.socket = None

        logging.basicConfig()
        self.logger = logging.getLogger('controller')
        self.logger.setLevel(logging.INFO)

        self._start_mrubis()

    def _start_mrubis(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sleep(1)
        self.socket.connect((self.host, self.port))
        self.logger.info('Connected to the Java side.')

    def get_from_mrubis(self, message):
        '''Send a message to mRUBiS and return the response as a dictionary'''
        self.println(message)
        self.logger.debug(f'Waiting for mRUBIS to answer to message {message}')
        return self.read_json()

    def readln(self):
        data = self.socket.recv(64000)
        message = data.decode('utf-8').strip("\n\r")

        try:
            json_message = json.loads(message)
            if json_message.get('number_of_chunks') is not None:
                return self.read_chunked(json_message.get('number_of_chunks'))
            else:
                return message
        except JSONDecodeError:
            return message

    def read_json(self):
        message = self.readln()
        try:
            json_message = json.loads(message)
        except JSONDecodeError:
            self.logger.error("Could not decode JSON input, received this:")
            self.logger.error(message)
            json_message = "not available"
        return json_message

    def read_chunked(self, num_chunks):
        self.println("send_chunks")

        message = ''

        for _ in range(num_chunks):
            while True:
                from_mrubis = self.readln()
                if from_mrubis != '':
                    message += from_mrubis
                    break
            self.println("received")

        from_mrubis = ''
        while True:
            from_mrubis = self.readln()
            if from_mrubis != '':
                break
        assert from_mrubis == "finished_sending"

        return message

    def println(self, message):
        self.socket.send(f"{message}\n".encode("utf-8"))

    def close_socket(self):
        self.socket.close()

    def send_exit_message(self):
        '''Tell mRUBiS to stop its main loop'''
        self.println("exit")
        _ = self.readln()
