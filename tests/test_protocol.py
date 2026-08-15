import socket
import threading
import time
import unittest

from infer_tab import format_detection_result
from tcp_client import RobotClient


class DetectionFormattingTests(unittest.TestCase):
    def test_result_is_ordered_by_class_id(self):
        result, readable = format_detection_result({0: "circle", 1: "square"}, [1, 0, 1])
        self.assertEqual(result, "circle,1,square,2")
        self.assertEqual(readable, "circle = 1  square = 2")

    def test_empty_result_is_none(self):
        self.assertEqual(format_detection_result({}, []), ("none", "ไม่พบวัตถุ"))

    def test_unsafe_model_class_is_rejected(self):
        with self.assertRaises(ValueError):
            format_detection_result({0: "bad,class"}, [0])


class TimeoutRecoveryTests(unittest.TestCase):
    def test_force_ready_disconnects_instead_of_reusing_socket(self):
        class TimeoutSocket:
            def __init__(self):
                self.sent = []
                self.send_event = threading.Event()

            def sendall(self, data):
                self.sent.append(data)
                self.send_event.set()

            def settimeout(self, _timeout):
                pass

            def recv(self, _size):
                time.sleep(0.01)
                raise socket.timeout()

            def close(self):
                pass

        fake_socket = TimeoutSocket()
        states = []
        callback_done = threading.Event()
        client = RobotClient(status_callback=states.append, done_timeout=5.0)
        client.sock = fake_socket
        client.connected = True
        try:
            client.send_detection_result("circle,1")
            self.assertTrue(fake_socket.send_event.wait(1.0))
            self.assertEqual(fake_socket.sent, [b"circle,1\n"])
            client.force_ready(done_callback=callback_done.set)
            self.assertTrue(callback_done.wait(2.0))
            self.assertFalse(client.connected)
            self.assertIsNone(client.sock)
            self.assertEqual(states[-1], "disconnected")
        finally:
            client.close_silently()


if __name__ == "__main__":
    unittest.main()
