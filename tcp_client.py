# -*- coding: utf-8 -*-
"""
tcp_client.py
-------------
TCP client for the Dobot MG400 robot arm. The AI program is the client, the robot
is the server.

Protocol implemented here:

  1. Handshake, once per connection:
     send "Ready" -> wait for "ACK". If the ACK does not arrive within ack_timeout
     the "Ready" message is sent again, up to ready_retries times.

  2. Detection result:
     send a string such as "circle,3,square,2", or "none" when nothing was found.

  3. Wait state:
     after a result is sent the client waits for "Done" from the robot. This wait has
     its own, much longer timeout (done_timeout) because the robot has to physically
     pick and place every part, which easily takes longer than a handshake.

     When done_timeout expires the client does NOT silently return to the ready state.
     Doing so would let the operator start another detection while the robot is still
     working, and the late "Done" would then be read as the answer to the next round,
     leaving the two sides permanently out of step. Instead the state becomes
     "wait_timeout" and the operator has to decide, via force_ready().

All reads go through _recv_line(), which keeps an internal buffer. That handles both
messages split across several packets and several messages merged into one packet.
Servers that do not terminate their messages with a newline (a plain Hercules setup,
for example) are handled too: a non-empty buffer that stops growing for quiet_gap
seconds is treated as one complete message.
"""

import socket
import threading
import time


class RobotClient:
    def __init__(
        self,
        log_callback=None,
        status_callback=None,
        connect_timeout=5.0,
        ack_timeout=5.0,
        done_timeout=180.0,
        ready_retries=5,
        quiet_gap=0.3,
    ):
        """
        log_callback(msg: str)      -> called for every message sent or received
        status_callback(state: str) -> called on every state change. States are
                                       'disconnected', 'connected', 'ready',
                                       'waiting_done' and 'wait_timeout'.
        connect_timeout             -> seconds allowed for the TCP connection itself
        ack_timeout                 -> seconds to wait for one "ACK" before re-sending
        done_timeout                -> seconds to wait for "Done" after a result
        ready_retries               -> how many times "Ready" may be sent
        quiet_gap                   -> seconds of silence after which an unterminated
                                       buffer counts as a complete message
        """
        self.sock = None
        self.connected = False
        self.lock = threading.Lock()
        self.log_callback = log_callback or (lambda msg: None)
        self.status_callback = status_callback or (lambda state: None)

        self.connect_timeout = connect_timeout
        self.ack_timeout = ack_timeout
        self.done_timeout = done_timeout
        self.ready_retries = ready_retries
        self.quiet_gap = quiet_gap

        self._buffer = b""
        self._abort_wait = False

    # ------------------------------------------------------------------ log ---
    def _log(self, direction, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_callback(f"[{ts}] {direction} {msg}")

    # --------------------------------------------------------------- receive ---
    def _take_line_from_buffer(self):
        """Pop one complete, non-empty message from the buffer, or return None."""
        while b"\n" in self._buffer:
            raw, self._buffer = self._buffer.split(b"\n", 1)
            text = raw.decode("utf-8", errors="ignore").strip()
            if text:
                return text
        return None

    def _flush_buffer(self):
        """Take the whole buffer as one message. Used for servers that do not
        terminate their messages, and when the peer disconnects mid-message."""
        text = self._buffer.decode("utf-8", errors="ignore").strip()
        self._buffer = b""
        return text or None

    def _recv_line(self, timeout, poll=0.5):
        """Read one message.

        Returns (status, text) where status is one of:
          'ok'       -> text holds the message
          'timeout'  -> nothing complete arrived within timeout seconds
          'aborted'  -> force_ready() asked us to stop waiting
          'closed'   -> the peer closed the connection
          'error'    -> socket error, text holds the description
        """
        deadline = time.time() + timeout
        last_data = None
        while True:
            text = self._take_line_from_buffer()
            if text is not None:
                return "ok", text

            # A server that never sends a newline still gets its message through:
            # once the buffer has been quiet for quiet_gap seconds, take what we have.
            if self._buffer and last_data is not None and (time.time() - last_data) > self.quiet_gap:
                text = self._flush_buffer()
                if text:
                    return "ok", text

            if self._abort_wait:
                return "aborted", ""

            remaining = deadline - time.time()
            if remaining <= 0:
                return "timeout", ""

            # While something unterminated is buffered, poll faster than quiet_gap,
            # otherwise the gap could not be noticed before the next blocking read.
            poll_now = min(poll, remaining)
            if self._buffer:
                poll_now = min(poll_now, self.quiet_gap)

            try:
                self.sock.settimeout(poll_now)
                chunk = self.sock.recv(4096)
            except socket.timeout:
                continue
            except OSError as e:
                text = self._flush_buffer()
                if text:
                    return "ok", text
                return "error", str(e)

            if not chunk:
                # The peer hung up. Anything already buffered is still a valid message.
                text = self._flush_buffer()
                if text:
                    return "ok", text
                return "closed", ""

            self._buffer += chunk
            last_data = time.time()

    def _drain(self):
        """Throw away anything still in flight so a late reply cannot be mistaken
        for the answer to the next request."""
        self._buffer = b""
        if not self.sock:
            return
        try:
            self.sock.settimeout(0.1)
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
        except OSError:
            pass

    # --------------------------------------------------------------- connect ---
    def _send_raw(self, text):
        if not self.sock:
            raise RuntimeError("socket is not connected")
        self.sock.sendall((text + "\n").encode("utf-8"))
        self._log("SEND", text)

    def connect(self, ip, port):
        """Open the connection and run the Ready/ACK handshake.

        Blocking: call this from a worker thread so the UI stays responsive.
        Returns (ok: bool, message: str).
        """
        self.close_silently()
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.connect_timeout)
            self.sock.connect((ip, int(port)))
        except Exception as e:
            self.sock = None
            self.connected = False
            self.status_callback("disconnected")
            self._log("ERR", f"เชื่อมต่อล้มเหลว: {e}")
            return False, str(e)

        self.connected = True
        self._buffer = b""
        self._abort_wait = False
        self._log("SYS", f"เชื่อมต่อสำเร็จไปยัง {ip}:{port}")
        self.status_callback("connected")
        return self._handshake()

    def _handshake(self):
        """Send "Ready" and wait for "ACK", re-sending until the retry budget is used up."""
        for attempt in range(1, self.ready_retries + 1):
            try:
                self._send_raw("Ready")
            except Exception as e:
                self.connected = False
                self.status_callback("disconnected")
                self._log("ERR", f"ส่ง Ready ไม่สำเร็จ: {e}")
                return False, str(e)

            status, data = self._recv_line(self.ack_timeout)

            if status == "ok":
                self._log("RECV", data)
                if data == "ACK":
                    self.status_callback("ready")
                    self._log("SYS", "ได้รับ ACK จากหุ่นยนต์ -- พร้อมทำงาน")
                    return True, "เชื่อมต่อและ handshake สำเร็จ (ได้รับ ACK)"
                self._log("SYS", f"ข้อความไม่ตรงกับ ACK ('{data}') -- ส่ง Ready ซ้ำ (ครั้งที่ {attempt + 1})")
            elif status == "timeout":
                self._log("SYS", f"ไม่ได้รับ ACK ใน {self.ack_timeout:.0f} วินาที -- ส่ง Ready ซ้ำ (ครั้งที่ {attempt + 1})")
            elif status == "closed":
                self.connected = False
                self.status_callback("disconnected")
                self._log("ERR", "หุ่นยนต์ปิดการเชื่อมต่อระหว่าง handshake")
                return False, "หุ่นยนต์ปิดการเชื่อมต่อระหว่าง handshake"
            else:
                self.connected = False
                self.status_callback("disconnected")
                self._log("ERR", f"ข้อผิดพลาดระหว่าง handshake: {data}")
                return False, data

        self.status_callback("connected")
        msg = f"ส่ง Ready ครบ {self.ready_retries} ครั้งแล้วยังไม่ได้รับ ACK"
        self._log("ERR", msg)
        return False, msg

    # ---------------------------------------------------------------- detect ---
    def send_detection_result(self, result_string, wait_for_done_callback=None):
        """Send the detection result, then wait for "Done" in a background thread.

        wait_for_done_callback(success: bool, message: str) is called once the wait
        finishes, whatever the outcome.
        """
        if not self.connected or not self.sock:
            if wait_for_done_callback:
                wait_for_done_callback(False, "ยังไม่ได้เชื่อมต่อกับหุ่นยนต์")
            return

        def _worker():
            with self.lock:
                self._abort_wait = False
                try:
                    self._send_raw(result_string)
                except Exception as e:
                    self.connected = False
                    self.status_callback("disconnected")
                    self._log("ERR", f"ส่งผลตรวจจับไม่สำเร็จ: {e}")
                    if wait_for_done_callback:
                        wait_for_done_callback(False, str(e))
                    return

                self.status_callback("waiting_done")
                self._log("SYS", f"เข้าสถานะรอ (Wait State) รอ Done ไม่เกิน {self.done_timeout:.0f} วินาที")

                deadline = time.time() + self.done_timeout
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        status, data = "timeout", ""
                    else:
                        status, data = self._recv_line(remaining)

                    if status == "ok":
                        self._log("RECV", data)
                        if data == "Done":
                            self.status_callback("ready")
                            if wait_for_done_callback:
                                wait_for_done_callback(True, "หุ่นยนต์ทำงานเสร็จแล้ว (Done) -- พร้อมตรวจจับรอบถัดไป")
                            return
                        # Anything that is not "Done" is chatter. Keep waiting for the
                        # real confirmation instead of ending the round early.
                        self._log("SYS", f"ข้ามข้อความที่ไม่ใช่ Done ('{data}') และรอต่อ")
                        continue

                    if status == "timeout":
                        self.status_callback("wait_timeout")
                        msg = (
                            f"หมดเวลารอ Done ({self.done_timeout:.0f} วินาที) "
                            "-- ยังค้างสถานะรอ กรุณาตรวจสอบหุ่นยนต์ แล้วกดปุ่มยกเลิกการรอถ้าต้องการเริ่มรอบใหม่"
                        )
                        self._log("ERR", msg)
                        if wait_for_done_callback:
                            wait_for_done_callback(False, msg)
                        return

                    if status == "aborted":
                        if wait_for_done_callback:
                            wait_for_done_callback(False, "ยกเลิกการรอ Done โดยผู้ใช้")
                        return

                    if status == "closed":
                        self.connected = False
                        self.status_callback("disconnected")
                        self._log("ERR", "หุ่นยนต์ปิดการเชื่อมต่อระหว่างรอ Done")
                        if wait_for_done_callback:
                            wait_for_done_callback(False, "หุ่นยนต์ปิดการเชื่อมต่อระหว่างรอ Done")
                        return

                    self.connected = False
                    self.status_callback("disconnected")
                    self._log("ERR", f"ข้อผิดพลาดระหว่างรอ Done: {data}")
                    if wait_for_done_callback:
                        wait_for_done_callback(False, data)
                    return

        threading.Thread(target=_worker, daemon=True).start()

    def force_ready(self, done_callback=None):
        """Operator override: stop waiting for "Done" and go back to ready.

        Runs in its own thread because it has to wait for the waiting worker to notice
        the abort flag and release the lock. Everything still queued on the socket is
        discarded, otherwise a late "Done" would answer the next detection.
        """
        def _worker():
            self._abort_wait = True
            with self.lock:
                self._drain()
                self._abort_wait = False
                if self.connected:
                    self.status_callback("ready")
                    self._log("SYS", "ยกเลิกการรอ Done -- ล้างข้อมูลค้างในสายแล้ว พร้อมตรวจจับรอบถัดไป")
                else:
                    self.status_callback("disconnected")
            if done_callback:
                done_callback()

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------ close ---
    def close_silently(self):
        """Drop any existing socket without touching the status indicator."""
        self._abort_wait = True
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.connected = False
        self._buffer = b""

    def close(self):
        was_open = self.sock is not None
        self.close_silently()
        self._abort_wait = False
        self.status_callback("disconnected")
        if was_open:
            self._log("SYS", "ปิดการเชื่อมต่อ")
