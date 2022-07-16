import os
import argparse
import sys 
import time

def kill_process(pid):
    print(f"kill pid {pid}")
    time.sleep(5)
    import signal
    os.kill(int(pid), signal.SIGKILL)


if __name__ =="__main__":
    if len(sys.argv) > 1:
        pid = sys.argv[1]
        kill_process(pid)
