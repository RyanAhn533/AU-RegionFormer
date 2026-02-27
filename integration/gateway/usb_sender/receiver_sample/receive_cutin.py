#!/usr/bin/env python3
"""
끼어들기(감정) USB 송신 수신부 샘플.
수신 프로토콜: 8바이트 감정 패킷 (SOF/TYPE/SEQ/LEN/Emotion&Status/Intensity/CRC/EOF).

[역계산 요약 - 개발자 참고]
  Byte 0: SOF 0xAA (검증만)
  Byte 1: TYPE 0x01 (검증/로그용)
  Byte 2: SEQ → 그대로 seq 값 (0~255)
  Byte 3: LEN → 반드시 2 (검증)
  Byte 4: emotion_code = (byte4 >> 4) & 0x0F
          stress = (byte4 >> 3) & 1, low_attention = (byte4 >> 2) & 1, drowsy = (byte4 >> 1) & 1
  Byte 5: emotion_intensity = (byte5 >> 5) & 0x07, status_intensity = (byte5 >> 2) & 0x07
  Byte 6: CRC = (B1+B2+B3+B4+B5) & 0xFF 로 검증
  Byte 7: EOF 0xFE (검증만)

수신부 환경:
  - Jetson(또는 송신 측)과 USB로 연결된 PC(Windows/Linux)에서
    가상 시리얼(COM / ttyUSB, ttyACM)로 수신합니다.
  - Windows: 장치 관리자 → 포트(COM & LPT) COM 번호 (예: COM3).
  - Linux: /dev/ttyUSB0, /dev/ttyACM0 등.

사용법:
  set CUTIN_RECEIVE_PORT=COM3
  python receive_cutin.py
  또는
  python receive_cutin.py COM3
"""
import os
import sys

PORT = os.environ.get("CUTIN_RECEIVE_PORT") or (sys.argv[1] if len(sys.argv) > 1 else "COM3")
BAUD = int(os.environ.get("CUTIN_SERIAL_BAUD", "115200"))

# 프로토콜 상수 (역계산 시 검증용)
SOF = 0xAA
EOF = 0xFE
PACKET_LEN = 8


def _read_packet(ser) -> bytes:
    """
    SOF(0xAA)로 프레임 동기화 후 정확히 8바이트를 읽어 반환합니다.
    동기화: SOF가 나올 때까지 1바이트씩 읽고, SOF 다음 7바이트를 추가로 읽음.
    """
    # SOF 찾기: 한 바이트씩 읽어서 0xAA가 나올 때까지 버림
    while True:
        b = ser.read(1)
        if not b:
            return b""
        if b[0] == SOF:
            break
    # SOF(1바이트) + 나머지 7바이트 = 총 8바이트
    rest = ser.read(7)
    if len(rest) < 7:
        return b""  # 타임아웃 등으로 부족하면 빈 패킷
    return bytes([SOF]) + rest


def _validate_and_parse(packet: bytes):
    """
    ********** 역계산 로직 (개발자 참고) **********
    - Byte 0: SOF 검증 (0xAA)
    - Byte 7: EOF 검증 (0xFE)
    - Byte 3: LEN == 2 검증
    - Byte 6: CRC8 = (Byte1+Byte2+Byte3+Byte4+Byte5) & 0xFF 검증
    - Byte 2: SEQ (그대로 사용)
    - Byte 4: [상위4bit] 감정코드 → (byte4 >> 4) & 0x0F
              [하위4bit] 상태플래그 → bit3=Stress, bit2=LowAttention, bit1=Drowsy
    - Byte 5: [상위3bit] 감정강도 → (byte5 >> 5) & 0x07
              [중간3bit] 상태강도 → (byte5 >> 2) & 0x07
    """
    try:
        from cutin_detection.usb_sender.protocol import parse_emotion_packet
    except ImportError:
        try:
            from ..protocol import parse_emotion_packet
        except ImportError:
            # 스크립트 단독 실행 시 (receiver_sample 폴더에서 실행)
            _dir = os.path.dirname(os.path.abspath(__file__))
            _usb_sender = os.path.dirname(_dir)
            _cutin_root = os.path.dirname(_usb_sender)
            if _cutin_root not in sys.path:
                sys.path.insert(0, _cutin_root)
            from usb_sender.protocol import parse_emotion_packet
    return parse_emotion_packet(packet)


def main():
    try:
        import serial
    except ImportError:
        print("Install: pip install pyserial")
        return 1
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
    except Exception as e:
        print(f"Failed to open {PORT}: {e}")
        print("Check cable and port (e.g. COM3, /dev/ttyUSB0)")
        return 1
    print(f"Listening on {PORT} @ {BAUD}. 8-byte emotion packet. Ctrl+C to stop.")
    try:
        while True:
            packet = _read_packet(ser)
            if not packet or len(packet) != PACKET_LEN:
                continue
            parsed = _validate_and_parse(packet)
            if parsed is None:
                # CRC/LEN/EOF 등 검증 실패 시 무시 (또는 로그)
                continue
            # 역계산 결과 출력 (끼어들기 테스트 시: cut_in이면 emotion_name="놀람")
            seq = parsed["seq"]
            emotion_name = parsed["emotion_name"]
            emotion_code = parsed["emotion_code"]
            ei = parsed["emotion_intensity"]
            si = parsed["status_intensity"]
            stress = parsed["stress"]
            low_att = parsed["low_attention"]
            drowsy = parsed["drowsy"]
            print(
                f"seq={seq} emotion={emotion_name}({emotion_code}) "
                f"intensity={ei}/{si} stress={stress} low_att={low_att} drowsy={drowsy}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
