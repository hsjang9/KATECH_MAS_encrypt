# 패키지
pip설치 시 - requirements_pip.txt   
conda 설치 시 - environment_conda.yaml   
설치 패키지 리스트 gymnasium numpy requests pyproj pandas pygame scipy matplotlib tqdm tensorboardx ipython pause pyglet cython pytorch pytorch-cuda   
   
# 실행 방법
python KATECH_MAS_main.py      
혹은 KATECH_MAS_SYN() 객체를 불러오고 run() 실행 (KATCH_MAS_main.py 참고)   
   
# 설정    
    katech_mas = KATECH_MAS_SYN() # 경로 생성 객체 생성   
    katech_mas.print = False      # 송신 response 출력 여부   
    katech_mas.display = False    # display 표출 여부   
    katech_mas.destination = 'E'  # E or W    
    katech_mas.period = 1         # 송신 주기 [s]   
    katech_mas.run()              # 경로 생성 SW 실행    

# 참고사항
- KIAPI PG만 지원   
- geofence 1개 대응 가능 (2개 이상 X)   
- geofence 위치는 왕복 4차로(긴 도로) 내에 존재해야 함   
- 차량 경로 생성은 왕복 4차로(긴 도로) 내에 있는 것을 기준으로 생성 함   
