import multiprocessing
import numpy as np


def _robot_control_worker(q, stop, ready, model_path):
    try:
        from orca_core import OrcaHand
        hand = OrcaHand(model_path)
        success, message = hand.connect()
        if not success:
            print(f"Robot process: Failed to connect: {message}")
            return
        hand.init_joints()
        ready.set()
        while not stop.is_set():
            try:
                angles = q.get(timeout=0.1)
                if angles:
                    joints = {n.split('_',1)[1] if n.startswith(('left_','right_')) else n: float(p) for n,p in angles.items()}
                    hand.set_joint_pos({n: np.rad2deg(p) for n,p in joints.items()})
            except Exception:
                continue
    except Exception as e:
        print(f"Robot process error: {e}")
    finally:
        try:
            hand.disable_torque()
            hand.disconnect()
        except Exception:
            pass


class RobotController:
    def __init__(self, model_path, ready_timeout=5.0):
        self._model_path = model_path
        self._ready_timeout = ready_timeout
        self._process = None
        self._queue = None
        self._stop_event = None

    def start(self):
        self._queue = multiprocessing.Queue()
        self._stop_event = multiprocessing.Event()
        ready_event = multiprocessing.Event()
        self._process = multiprocessing.Process(
            target=_robot_control_worker,
            args=(self._queue, self._stop_event, ready_event, self._model_path),
            daemon=True)
        self._process.start()
        if not ready_event.wait(timeout=self._ready_timeout):
            self._process.terminate()
            raise TimeoutError("Robot initialization timeout")

    def send_angles(self, angles):
        if self._queue is not None:
            self._queue.put_nowait(angles)

    def stop(self):
        if self._process is None or not self._process.is_alive():
            return
        self._stop_event.set()
        self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
