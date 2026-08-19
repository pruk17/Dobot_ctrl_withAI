# -*- coding: utf-8 -*-
"""
mock_robot_server.py
--------------------
Stand-in for the Dobot MG400 (which is the TCP server) so the AI program can be
tested without the real robot. An alternative to Hercules.

Behaviour:
  - listen on the given host and port
  - reply "ACK" to "Ready"
  - reply "Done" to anything else, after work_seconds of pretend work

The accept loop restarts after every disconnect, so the program does not have to be
started again each time the client reconnects. Stop it with Ctrl+C.

Usage:
    python mock_robot_server.py --host 0.0.0.0 --port 29999
    python mock_robot_server.py --work-seconds 15   # to test the Done timeout path
"""

import argparse
import socket
import time


def serve_client(conn, addr, work_seconds):
    print(f"[mock] client connected from {addr}")
    # A one second timeout keeps the blocking read short. On Windows a Python process
    # sitting inside a blocking socket call cannot notice Ctrl+C until that call
    # returns, so without this the only way to stop the server would be to kill it.
    conn.settimeout(1.0)
    buffer = b""
    while True:
        try:
            chunk = conn.recv(1024)
        except socket.timeout:
            continue
        except OSError as e:
            print(f"[mock] socket error: {e}")
            return
        if not chunk:
            print("[mock] client closed the connection")
            return

        buffer += chunk
        # Accept messages with or without a trailing newline.
        if b"\n" in buffer:
            lines, buffer = buffer.split(b"\n")[:-1], buffer.split(b"\n")[-1]
        else:
            lines, buffer = [buffer], b""

        for raw in lines:
            data = raw.decode("utf-8", errors="ignore").strip()
            if not data:
                continue
            print(f"[mock] received: {data}")

            if data == "Ready":
                conn.sendall(b"ACK\n")
                print("[mock] sent: ACK")
            else:
                # Anything else is treated as a detection result.
                print(f"[mock] pretending to work for {work_seconds} seconds ...")
                time.sleep(work_seconds)
                conn.sendall(b"Done\n")
                print("[mock] sent: Done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6001)
    parser.add_argument(
        "--work-seconds", type=float, default=2.0,
        help="how long the fake robot pretends to work before answering Done",
    )
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    # Same reason as in serve_client: accept() has to return regularly, otherwise
    # Ctrl+C is not processed while the server waits for a client.
    server.settimeout(1.0)
    print(f"[mock] listening on {args.host}:{args.port} (Ctrl+C to stop)")

    try:
        while True:
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            try:
                serve_client(conn, addr, args.work_seconds)
            finally:
                conn.close()
            print("[mock] waiting for the next connection ...")
    except KeyboardInterrupt:
        print("\n[mock] stopped")
    finally:
        server.close()


if __name__ == "__main__":
    main()
