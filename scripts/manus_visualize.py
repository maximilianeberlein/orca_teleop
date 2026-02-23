"""Visualize raw Manus glove skeleton data in the 3D viewer — no retargeting, no robot."""

import sys
import time
import argparse
from orca_teleop import ManusIngress


def main():
    parser = argparse.ArgumentParser(description='Visualize raw Manus glove skeleton in 3D viewer')
    parser.add_argument('model_path', help='Path to OrcaHand model directory')
    parser.add_argument('urdf_path', help='Path to URDF file for viewer')
    parser.add_argument('--glove-id', required=True, help='Manus glove hex ID (e.g. 1bd24715)')
    parser.add_argument('--zmq-addr', default='tcp://localhost:8000', help='ZMQ address for Manus SDK stream')
    args = parser.parse_args()

    try:
        from orca_teleop.viewer import URDFViewer
        viewer = URDFViewer(args.urdf_path)
    except ImportError:
        print("viser not installed — required for this script (pip install -e '.[viewer]')")
        return 1
    except Exception as e:
        print(f"Failed to start viewer: {e}")
        return 1

    def on_skeleton(data):
        viewer.update_mano_points(data["skeleton"][:, :3])

    ingress = ManusIngress(args.model_path, args.glove_id, callback=on_skeleton,
                           zmq_addr=args.zmq_addr)
    ingress.start()
    print("Manus skeleton viewer running. Ctrl+C to quit.")

    try:
        while not viewer.stopped:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        ingress.cleanup()
        viewer.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
