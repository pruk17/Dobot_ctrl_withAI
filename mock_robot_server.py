# -*- coding: utf-8 -*-
"""
mock_robot_server.py
----------------------
สคริปต์จำลองฝั่งหุ่นยนต์ Dobot MG400 (ทำหน้าที่เป็น TCP Server) สำหรับใช้ทดสอบโปรแกรม AI
โดยไม่ต้องมีหุ่นยนต์จริง (ทางเลือกแทน Hercules)

พฤติกรรม:
  - เปิด listen ที่ IP/Port ที่กำหนด
  - เมื่อรับข้อความ "Ready" -> ตอบกลับ "ACK"
  - เมื่อรับข้อความผลตรวจจับ (เช่น "circle,3,square,2" หรือ "none") -> รอสักครู่ (จำลองการทำงานของหุ่นยนต์)
    แล้วตอบกลับ "Done"

วิธีใช้:
    python mock_robot_server.py --host 0.0.0.0 --port 29999
"""

import argparse
import socket
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29999)
    parser.add_argument("--work-seconds", type=float, default=2.0, help="เวลาจำลองที่หุ่นยนต์ใช้ทำงานก่อนตอบ Done")
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[mock_robot_server] กำลังรอการเชื่อมต่อที่ {args.host}:{args.port} ...")

    conn, addr = server.accept()
    print(f"[mock_robot_server] เชื่อมต่อจาก {addr}")

    try:
        while True:
            data = conn.recv(1024).decode("utf-8", errors="ignore").strip()
            if not data:
                print("[mock_robot_server] การเชื่อมต่อถูกปิด")
                break
            print(f"[mock_robot_server] ได้รับ: {data}")

            if data == "Ready":
                conn.sendall(b"ACK\n")
                print("[mock_robot_server] ส่ง: ACK")
            else:
                # สมมติว่าข้อความอื่น ๆ คือผลตรวจจับ เช่น circle,3,square,2 หรือ none
                print(f"[mock_robot_server] กำลังจำลองการทำงานของหุ่นยนต์ ({args.work_seconds} วินาที)...")
                time.sleep(args.work_seconds)
                conn.sendall(b"Done\n")
                print("[mock_robot_server] ส่ง: Done")
    finally:
        conn.close()
        server.close()


if __name__ == "__main__":
    main()
