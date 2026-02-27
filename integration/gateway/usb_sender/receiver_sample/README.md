# 끼어들기(감정) USB 수신부 샘플

## 통신 방식

- **가상 시리얼(CDC ACM, COM 포트)** 방식입니다.
- 송신 측(Jetson 등)이 **500ms마다 고정 8바이트 바이너리 패킷**을 시리얼로 보내면, 이 수신부가 SOF 동기화 후 8바이트를 읽고 검증·역계산해 터미널에 출력합니다.

## 전송 프로토콜 요약 (8바이트)

| 바이트 | 필드 | 값/의미 |
|--------|------|---------|
| 0 | SOF | 0xAA (시작) |
| 1 | TYPE | 0x01 (감정 패킷) |
| 2 | SEQ | 0~255 (순서, 순환) |
| 3 | LEN | 2 (데이터 길이) |
| 4 | Emotion & Status | 상위4bit: 감정(0~5). 하위4bit: Stress/LowAttention/Drowsy/Reserved |
| 5 | Intensity | 상위3bit: 감정강도(0~7). 중간3bit: 상태강도(0~7). 하위2bit: Reserved |
| 6 | CRC8 | (Byte1+2+3+4+5) & 0xFF |
| 7 | EOF | 0xFE (종료) |

- **역계산:** 감정 코드 = `(Byte4>>4)&0x0F`, 상태 플래그 = Byte4 하위 4bit, 감정/상태 강도 = Byte5에서 비트 시프트로 추출.  
  상세는 `usb_sender/protocol.py`의 `parse_emotion_packet()` 및 파일 상단 주석 참고.

## 수신부 환경 (기본)

- **Jetson과 USB로 연결된 PC(Windows 또는 Linux)** 에서 COM 포트(Windows) 또는 ttyUSB/ttyACM(Linux)로 수신하는 것을 전제로 합니다.
- 다른 장치(MCU, ECU 등)에서 수신할 경우 **동일 8바이트 프로토콜**로 받으면 됩니다 (SOF 동기화 → 8바이트 읽기 → CRC 검증 → Byte4/5 역계산).

## 역할

Jetson(또는 송신 측)에서 500ms마다 보내는 **8바이트 감정 패킷**을 수신해, SEQ·감정명·강도·상태 플래그를 터미널에 출력합니다.  
현재 송신 측은 끼어들기 감지 시 감정만 "놀람(1)"로 설정하고 나머지는 0입니다.

## 준비

- PC와 Jetson을 USB 시리얼로 연결 (가상 COM 포트 또는 USB‑UART 어댑터)
- PC에 Python + pyserial 설치: `pip install pyserial`

## 포트 확인

- **Windows:** 장치 관리자 → 포트(COM & LPT) → 예: COM3
- **Linux:** `ls /dev/ttyUSB*` 또는 `ls /dev/ttyACM*`

## 실행

```bash
# 포트를 인자로
python receive_cutin.py COM3

# 또는 환경변수
set CUTIN_RECEIVE_PORT=COM3   # Windows
export CUTIN_RECEIVE_PORT=/dev/ttyUSB0   # Linux
python receive_cutin.py
```

보드레이트가 다르면: `set CUTIN_SERIAL_BAUD=9600` 등으로 맞춥니다.

## 출력 예

```
seq=0 emotion=공포(0) intensity=0/0 stress=False low_att=False drowsy=False
seq=1 emotion=놀람(1) intensity=0/0 stress=False low_att=False drowsy=False
seq=2 emotion=공포(0) intensity=0/0 stress=False low_att=False drowsy=False
```

(끼어들기 감지 시점에만 `emotion=놀람(1)` 로 나옵니다.)
