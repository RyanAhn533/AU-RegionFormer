"""
500ms 단위로 전역 끼어들기 상태를 USB 시리얼로 전송.
별도 스레드에서 구동. 전역변수는 core.cutin_state에서 읽음.

전송 형식: 감정 패킷 8바이트 (SOF/TYPE/SEQ/LEN/Emotion&Status/Intensity/CRC/EOF)
- 끼어들기 감지 시 → Byte4 감정만 "놀람(1)", 나머지는 기본값 0.
"""
import os
import time
import threading

SERIAL_PORT = os.environ.get("CUTIN_SERIAL_PORT", "COM3")   # Windows: COM3 등 / Linux: /dev/ttyUSB0
SERIAL_BAUD = int(os.environ.get("CUTIN_SERIAL_BAUD", "115200"))
INTERVAL_SEC = 0.5  # 500ms


def run_usb_sender_500ms(shutdown_event=None, port=None, baud=None, interval_sec=INTERVAL_SEC):
    """
    스레드 진입점: interval_sec마다 전역 끼어들기 상태를 읽어
    감정 패킷 8바이트를 생성한 뒤 시리얼로 전송합니다.

    데이터 매핑 (현재):
      - cut_in == True  → Byte4 감정 = 놀람(1), 나머지 0
      - cut_in == False → Byte4 감정 = 0, 나머지 0
    SEQ는 매 전송마다 0~255 순환 증가합니다.
    """
    port = port or SERIAL_PORT
    baud = baud or SERIAL_BAUD
    try:
        import serial
    except ImportError:
        print("[USB Sender] pyserial not installed. pip install pyserial")
        return
    try:
        from cutin_detection.core.cutin_state import get_state
    except ImportError:
        try:
            from ..core.cutin_state import get_state
        except ImportError:
            from core.cutin_state import get_state
    try:
        from cutin_detection.usb_sender.protocol import build_packet_from_cutin
    except ImportError:
        from .protocol import build_packet_from_cutin

    if shutdown_event is None:
        class Dummy:
            def is_set(self): return False
        shutdown_event = Dummy()

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except Exception as e:
        print(f"[USB Sender] Failed to open {port}: {e}")
        return
    print(f"[USB Sender] Opened {port} @ {baud}, sending 8-byte emotion packet every {interval_sec}s")
    # 시퀀스 번호: 0~255 순환 (패킷 손실 검출용)
    seq = 0
    try:
        while not shutdown_event.is_set():
            t0 = time.monotonic()
            # 전역 상태에서 끼어들기 감지 여부만 사용
            detected, _, _, _, _ = get_state()
            # 끼어들기 감지 → 놀람(1), 미감지 → 0, 나머지 기본값 0
            packet = build_packet_from_cutin(cut_in_detected=detected, seq=seq)
            try:
                ser.write(packet)
            except Exception as e:
                print(f"[USB Sender] write error: {e}")
                break
            seq = (seq + 1) & 0xFF
            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, interval_sec - elapsed)
            if sleep_time > 0:
                shutdown_event.wait(timeout=sleep_time) if hasattr(shutdown_event, "wait") else time.sleep(sleep_time)
    finally:
        try:
            ser.close()
        except Exception:
            pass
        print("[USB Sender] Stopped.")
