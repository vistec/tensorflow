import tensorflow as tf
import os
import numpy as np
import threading
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("stop from id: %d\n" % worker_id)
            coord.request_stop()
        else:
            print("working on id: %d\n " % worker_id)
        time.sleep(1)

coord = tf.train.Coordinator()

threads = [threading.Thread(target=MyLoop,args=(coord,i,)) for i in range(5)]
for t in threads:
    t.start()
coord.join()

