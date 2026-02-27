"""
감정 패킷 8바이트 프로토콜 정의 및 패킷 생성/역계산 유틸.

프로토콜 개요:
  Byte 0: SOF 0xAA (시작)
  Byte 1: TYPE 0x01 (감정 패킷)
  Byte 2: SEQ 0~255 (순서, 패킷 손실 검출)
  Byte 3: LEN 2 (데이터 길이 = Byte 4,5)
  Byte 4: [상위4bit] 감정코드(0~5) | [하위4bit] 상태플래그(Stress, LowAttention, Drowsy, Reserved)
  Byte 5: [상위3bit] 감정강도(0~7) | [중간3bit] 상태강도(0~7) | [하위2bit] Reserved
  Byte 6: CRC8 = (Byte1+Byte2+Byte3+Byte4+Byte5) & 0xFF
  Byte 7: EOF 0xFE (종료)
"""
from typing import List, Tuple, Optional

# ========== 프로토콜 상수 (실제 적용 시 이 값으로 고정) ==========
SOF = 0xAA  # Start Of Frame: 패킷 시작 알림
EOF = 0xFE  # End Of Frame: 패킷 종료 알림
TYPE_EMOTION = 0x01  # 감정 패킷 (추후 확장 가능)
DATA_LEN = 2  # 실제 데이터 바이트 수 (Byte 4, 5)

# 감정 코드 정의 (Byte 4 상위 4bit)
EMOTION_FEAR = 0       # 공포
EMOTION_SURPRISE = 1   # 놀람
EMOTION_ANGER = 2      # 분노
EMOTION_SADNESS = 3    # 슬픔
EMOTION_HAPPY = 4      # 행복
EMOTION_DISGUST = 5    # 혐오

EMOTION_NAMES = {
    0: "공포",
    1: "놀람",
    2: "분노",
    3: "슬픔",
    4: "행복",
    5: "혐오",
}

# 상태 플래그 비트 위치 (Byte 4 하위 4bit)
# Bit 3: Stress (1=고긴장)
# Bit 2: Low Attention (1=주의산만)
# Bit 1: Drowsy (1=졸음후보)
# Bit 0: Reserved (0)
STATUS_STRESS = 0x08       # 1 << 3
STATUS_LOW_ATTENTION = 0x04  # 1 << 2
STATUS_DROWSY = 0x02       # 1 << 1
# STATUS_RESERVED = 0x01    # 사용 안 함


def build_emotion_packet(
    emotion_code: int,
    stress: bool = False,
    low_attention: bool = False,
    drowsy: bool = False,
    emotion_intensity: int = 0,
    status_intensity: int = 0,
    seq: int = 0,
) -> bytes:
    """
    감정 패킷 8바이트를 생성합니다.
    개발 시 참고: 실제 데이터가 수집되면 이 함수 인자만 확장하면 됩니다.

    Args:
        emotion_code: 감정 코드 0~5 (0:공포, 1:놀람, 2:분노, 3:슬픔, 4:행복, 5:혐오)
        stress: 고긴장 플래그 (Byte4 bit3)
        low_attention: 주의산만 플래그 (Byte4 bit2)
        drowsy: 졸음후보 플래그 (Byte4 bit1)
        emotion_intensity: 감정 강도 0~7 (Byte5 상위 3bit)
        status_intensity: 상태 강도 0~7 (Byte5 중간 3bit)
        seq: 시퀀스 번호 0~255 (Byte2, 순환)

    Returns:
        8바이트 패킷 (bytes)
    """
    # 시퀀스는 0~255만 유효 (한 바이트)
    seq = seq & 0xFF
    emotion_code = max(0, min(5, emotion_code)) & 0x0F
    emotion_intensity = max(0, min(7, emotion_intensity)) & 0x07
    status_intensity = max(0, min(7, status_intensity)) & 0x07

    # ----- Byte 4: [상위4bit] 감정코드 | [하위4bit] 상태플래그 -----
    # 비트 결합: (Code << 4) | (Flag)
    # 예: 놀람(1) + 졸음(1) → (1<<4) | 0b0010 = 0x12
    status_flags = 0
    if stress:
        status_flags |= STATUS_STRESS
    if low_attention:
        status_flags |= STATUS_LOW_ATTENTION
    if drowsy:
        status_flags |= STATUS_DROWSY
    byte4 = (emotion_code << 4) | status_flags

    # ----- Byte 5: [상위3bit] 감정강도 | [중간3bit] 상태강도 | [하위2bit] 0 -----
    # 비트 결합: (감정강도 << 5) | (상태강도 << 2)
    byte5 = (emotion_intensity << 5) | (status_intensity << 2)

    # ----- CRC8: (Byte1 + Byte2 + Byte3 + Byte4 + Byte5) & 0xFF -----
    # 합계가 300(0x12C)이면 하위 8bit만 사용 → 0x2C
    crc = (TYPE_EMOTION + seq + DATA_LEN + byte4 + byte5) & 0xFF

    packet = bytes([
        SOF,           # Byte 0
        TYPE_EMOTION,  # Byte 1
        seq,           # Byte 2
        DATA_LEN,      # Byte 3
        byte4,         # Byte 4
        byte5,        # Byte 5
        crc,           # Byte 6
        EOF,           # Byte 7
    ])
    return packet


def build_packet_from_cutin(cut_in_detected: bool, seq: int = 0) -> bytes:
    """
    현재는 감정/상태 데이터가 완전히 수집되지 않아,
    끼어들기 감지만으로 패킷을 만듭니다.

    - 끼어들기 감지 시 → Byte4 감정만 "놀람(1)" 로 설정
    - 나머지(상태 플래그, 감정/상태 강도)는 모두 기본값 0

    Args:
        cut_in_detected: 끼어들기 감지 여부 (True → 놀람)
        seq: 시퀀스 번호 0~255

    Returns:
        8바이트 패킷 (bytes)
    """
    emotion_code = EMOTION_SURPRISE if cut_in_detected else 0
    return build_emotion_packet(
        emotion_code=emotion_code,
        stress=False,
        low_attention=False,
        drowsy=False,
        emotion_intensity=0,
        status_intensity=0,
        seq=seq,
    )


def parse_emotion_packet(packet: bytes) -> Optional[dict]:
    """
    수신한 8바이트 패킷을 역계산하여 감정/상태/강도/SEQ 등을 반환합니다.
    SOF/EOF/CRC/LEN 검증 후 파싱합니다.

    Args:
        packet: 정확히 8바이트

    Returns:
        검증 성공 시 dict: emotion_code, emotion_name, stress, low_attention, drowsy,
                          emotion_intensity, status_intensity, seq, type_id
        검증 실패 시 None
    """
    if packet is None or len(packet) != 8:
        return None

    # ----- 1) SOF / EOF 검증 -----
    if packet[0] != SOF:
        return None
    if packet[7] != EOF:
        return None

    # ----- 2) LEN 검증 (Byte3 == 2) -----
    if packet[3] != DATA_LEN:
        return None

    # ----- 3) CRC8 검증: Byte6 == (Byte1+2+3+4+5) & 0xFF -----
    expected_crc = (packet[1] + packet[2] + packet[3] + packet[4] + packet[5]) & 0xFF
    if packet[6] != expected_crc:
        return None

    # ----- 4) Byte 2: SEQ -----
    seq = packet[2] & 0xFF

    # ----- 5) Byte 4 역계산: [상위4bit] 감정코드 | [하위4bit] 상태플래그 -----
    byte4 = packet[4]
    emotion_code = (byte4 >> 4) & 0x0F
    stress = bool((byte4 >> 3) & 1)
    low_attention = bool((byte4 >> 2) & 1)
    drowsy = bool((byte4 >> 1) & 1)
    # bit0 Reserved 는 사용 안 함

    # ----- 6) Byte 5 역계산: [상위3bit] 감정강도 | [중간3bit] 상태강도 -----
    byte5 = packet[5]
    emotion_intensity = (byte5 >> 5) & 0x07
    status_intensity = (byte5 >> 2) & 0x07
    # 하위 2bit Reserved

    return {
        "type_id": packet[1],
        "seq": seq,
        "emotion_code": emotion_code,
        "emotion_name": EMOTION_NAMES.get(emotion_code, f"Unknown({emotion_code})"),
        "stress": stress,
        "low_attention": low_attention,
        "drowsy": drowsy,
        "emotion_intensity": emotion_intensity,
        "status_intensity": status_intensity,
    }


def find_packet_sync(stream: bytes, start: int = 0) -> int:
    """
    바이트 스트림에서 SOF(0xAA) 위치를 찾습니다.
    수신부에서 프레임 동기화할 때 사용합니다.

    Args:
        stream: 수신 버퍼
        start: 검색 시작 인덱스

    Returns:
        SOF 인덱스. 없으면 -1
    """
    try:
        idx = stream.index(SOF, start)
        return idx
    except ValueError:
        return -1
