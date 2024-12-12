import signal
import sys

from KATECH_MAS_run import KATECH_MAS_SYN

sys.path.append('./')

def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    katech_mas = KATECH_MAS_SYN()
    katech_mas.prints = True       # 송신 response 출력 여부
    katech_mas.display = True      # display 표출 여부
    katech_mas.send_prints = False # 송신 send data prints
    katech_mas.destination = 'E'   # E or W 
    katech_mas.period = 1          # 송신 주기 [s]
    katech_mas.run()               # 경로 생성 실행

