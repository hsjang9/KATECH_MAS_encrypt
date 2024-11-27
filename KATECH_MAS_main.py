import signal
import sys

from KATECH_MAS_run import KATECH_MAS_SYN

def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    katech_mas = KATECH_MAS_SYN() # 경로 생성 객체 생성
    katech_mas.prints = False      # 송신 response 출력 여부
    katech_mas.display = False    # display 표출 여부
    katech_mas.destination = 'E'  # E or W 
    katech_mas.period = 1         # 송신 주기 [s]
    katech_mas.run()              # 경로 생성 SW 실행

# KIAPI PG만 지원
# geofence 1개 대응 가능 (2개 이상 X)
# geofence 위치는 왕복 4차로(긴 도로) 내에 존재해야 함
# 차량 경로 생성은 왕복 4차로(긴 도로) 내에 있는 것을 기준으로 생성 함
