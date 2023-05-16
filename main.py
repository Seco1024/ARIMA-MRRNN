import threading
import subprocess

def run_script(script):
    subprocess.run(['/bin/bash', script])

thread1 = threading.Thread(target=run_script, args=('run_VARMA1.sh',))
thread2 = threading.Thread(target=run_script, args=('run_VARMA2.sh',))

thread1.start()
thread2.start()

thread1.join()
thread2.join()