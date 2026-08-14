# -*- coding: utf-8 -*-
"""
tcp_client.py
--------------
โปรแกรม AI ทำหน้าที่เป็น TCP Client เชื่อมต่อไปยังแขนหุ่นยนต์ Dobot MG400 (TCP Server)
ตามข้อกำหนด:
  1. เชื่อมต่อสำเร็จ -> ส่งข้อความ "Ready" และรอรับ "ACK" จากหุ่นยนต์ก่อนเริ่มทำงาน (ทำครั้งเดียวตอนเริ่มต้น)
  2. เมื่อพบวัตถุ -> ส่งผลลัพธ์ในรูปแบบ String เช่น "circle,3,square,2" หรือ "none" ถ้าไม่พบวัตถุ
  3. หลังส่งผลลัพธ์ -> เข้าสู่สถานะรอ (Wait State) และรอรับ "Done" จากหุ่นยนต์ก่อนจะพร้อมตรวจจับรอบถัดไป
"""

import socket
import threading
import time


class RobotClient:
    def __init__(self, log_callback=None, status_callback=None):
        """
        log_callback(msg: str)      -> เรียกทุกครั้งที่มีข้อความสื่อสาร (สำหรับแสดงใน Communication Log)
        status_callback(state: str) -> เรียกเมื่อสถานะเปลี่ยน เช่น 'connected', 'ready', 'waiting_done', 'idle', 'disconnected'
        """
        self.sock = None
        self.connected = False
        self.lock = threading.Lock()
        self.log_callback = log_callback or (lambda msg: None)
        self.status_callback = status_callback or (lambda state: None)
        self.recv_timeout = 10.0  # วินาที: เวลารอรับ ACK / Done ก่อน timeout

    def _log(self, direction, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_callback(f"[{ts}] {direction} {msg}")

    def connect(self, ip, port, timeout=5.0):
        """เชื่อมต่อไปยังหุ่นยนต์ และทำ handshake Ready/ACK ครั้งเดียวตอนเริ่มต้น"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect((ip, int(port)))
            self.connected = True
            self._log("SYS", f"เชื่อมต่อสำเร็จไปยัง {ip}:{port}")
            self.status_callback("connected")

            # ส่ง Ready และรอรับ ACK (ทำครั้งเดียวตอนเริ่มต้น)
            self._send_raw("Ready")
            self.sock.settimeout(self.recv_timeout)
            data = self.sock.recv(1024).decode("utf-8", errors="ignore").strip()
            self._log("RECV", data if data else "(ไม่มีข้อมูล)")
            if data == "ACK":
                self.status_callback("ready")
                self._log("SYS", "ได้รับ ACK จากหุ่นยนต์ -- พร้อมทำงาน")
                return True, "เชื่อมต่อและ handshake สำเร็จ (ได้รับ ACK)"
            else:
                self.status_callback("connected")
                return True, f"เชื่อมต่อสำเร็จ แต่ได้รับข้อความไม่ตรงกับ ACK: '{data}'"
        except Exception as e:
            self.connected = False
            self.status_callback("disconnected")
            self._log("ERR", f"เชื่อมต่อล้มเหลว: {e}")
            return False, str(e)

    def _send_raw(self, text):
        if not self.sock:
            raise RuntimeError("ยังไม่ได้เชื่อมต่อ")
        self.sock.sendall((text + "\n").encode("utf-8"))
        self._log("SEND", text)

    def send_detection_result(self, result_string, wait_for_done_callback=None):
        """
        ส่งผลลัพธ์การตรวจจับ (เช่น 'circle,3,square,2' หรือ 'none') ไปยังหุ่นยนต์
        จากนั้นเข้าสถานะรอ (Wait State) และรอรับ 'Done' แบบ non-blocking (รันใน thread แยก)
        wait_for_done_callback(success: bool, message: str) จะถูกเรียกเมื่อได้รับ Done หรือเกิด timeout/error
        """
        if not self.connected or not self.sock:
            if wait_for_done_callback:
                wait_for_done_callback(False, "ยังไม่ได้เชื่อมต่อกับหุ่นยนต์")
            return

        def _worker():
            try:
                with self.lock:
                    self._send_raw(result_string)
                    self.status_callback("waiting_done")
                    self.sock.settimeout(self.recv_timeout)
                    data = self.sock.recv(1024).decode("utf-8", errors="ignore").strip()
                self._log("RECV", data if data else "(ไม่มีข้อมูล)")
                if data == "Done":
                    self.status_callback("ready")
                    if wait_for_done_callback:
                        wait_for_done_callback(True, "หุ่นยนต์ทำงานเสร็จแล้ว (Done) -- พร้อมตรวจจับรอบถัดไป")
                else:
                    self.status_callback("ready")
                    if wait_for_done_callback:
                        wait_for_done_callback(False, f"ได้รับข้อความไม่ตรงกับ Done: '{data}'")
            except socket.timeout:
                self.status_callback("ready")
                self._log("ERR", "หมดเวลารอ 'Done' จากหุ่นยนต์ (timeout)")
                if wait_for_done_callback:
                    wait_for_done_callback(False, "หมดเวลารอ 'Done' จากหุ่นยนต์ (timeout)")
            except Exception as e:
                self.status_callback("disconnected")
                self.connected = False
                self._log("ERR", f"ข้อผิดพลาดระหว่างส่ง/รับข้อมูล: {e}")
                if wait_for_done_callback:
                    wait_for_done_callback(False, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.connected = False
        self.status_callback("disconnected")
        self._log("SYS", "ปิดการเชื่อมต่อ")
