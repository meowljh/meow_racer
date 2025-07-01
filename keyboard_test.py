import keyboard
import time, sys
import threading
# from pynput import keyboard

def on_space():
    print('space was pressed')
 
def read_event(status):
    event = keyboard.read_event() 
    if event.event_type == keyboard.KEY_DOWN:
        if event.name == 'p':
            status.append("PAUSE")
        elif event.name == 's':
            status.append('START')
        elif event.name == 'q':
            status.append('QUIT')
        
def print_key_event(e):
    print(f"Key {e.name} was {e.event_type}")
    
# keyboard.hook(print_key_event)
# keyboard.wait("esc")
# keyboard.add_hotkey(
#     'space', on_space
# )
# keyboard.wait()
def load_thread(status_list):
    tr = threading.Thread(target=read_event, name="check", args=(status_list,))
    tr.daemon = True
    tr.start()
    return tr

prev_stat = 1
status_list = []
tr = load_thread(status_list)
while True:
    ## do things.. ##
    print("status list ", status_list)
    for e in status_list[::-1]:
        if e == 'QUIT':
            prev_stat = -1
            sys.exit()
        elif e == 'PAUSE':
            status_list = []
            prev_stat = 0
            print("PAUSE for 1 seconds") 
            time.sleep(1)
            tr.join() 
            tr = load_thread(status_list) 
            
        elif e == 'START' and prev_stat == 0:
            prev_stat = 1
            print('resume..')
            

 