## Troubleshooting

### Installing `mujoco_py`

#### Windows10 환경에서의 설치 방법

[참고] https://medium.com/@sayanmndl21/install-openai-gym-with-box2d-and-mujoco-in-windows-10-e25ee9b5c1d5
https://talkingaboutme.tistory.com/entry/RL-Windows-10-mujoco-py


#### Error - 1
- 문제 상황: `python -c "mujoco_py"` 명령어로 본래 c++ engine인 MuJoCo를 compile하는 과정에서 아래 에러 발생
- 에러: `ImportError: DLL load failed while importing cymj: 지정된 모듈을 찾을 수 없습니다.`
- 해결 방법: https://github.com/openai/mujoco-py/issues/638
```py
os.add_dll_directory(r"C:\Users\7459985\.mujoco\mujoco210\bin".replace("\\", "/"))
##위의 코드 한줄을 mujoco-py/setup.py의 상단에 추가해 주어야 했음.
```




#### Error - 2
- 문제 상황: 전부 compile과 install까지 하고 rendering단에서 `OpenGL`이 없기 때문에 에러 발생
- 에러: `ERROR: GLEW initialization error: Missing GL version`
- 해결 방법: GLEW가 설치가 안되어 있는 것이기 때문에 설치해서 사용자 환경 변수에 PATH 추가해 주면 해결 되는 문제였다.



## Datasets?
- `halfcheetah-medium-expert-v2`
- `hopper-medium-expert-v2`
- `ant-medium-expert-v2`
- `walker2d-medium-expert-v2`
