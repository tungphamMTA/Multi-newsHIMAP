from queue import Queue
import requests
import time 
import os 
from flask import Flask,jsonify
from flask import request
from flask_cors import CORS
# from subprocess import Popen, CREATE_NEW_CONSOLE
from multiprocessing import Process

app = Flask(__name__)
CORS(app)

def run():
    os.system("bash ./activate_env.bash")

@app.route('/change_status_root', methods=['POST'])
def post():
    content = request.get_json()
    url = 'http://127.0.0.1:6687/change_status'
    myobj = {'status': content["status"]}
    try:
        requests.post(url=url, json=myobj)
    except:
        pass
    time.sleep(5)
    if content["status"] ==False:
        p=Process(target=run, args=())
        p.start()
        # subprocess.call('python app.py', shell=True)

    return {"result":True} 

app.run(host='0.0.0.0', port=5002)

