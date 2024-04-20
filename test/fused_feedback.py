import time
import numpy as np
from qrac.feedback import CrazyflieFeedback, OptiTrackFeedback

def main():
    runtime = 5
    pose = OptiTrackFeedback(frame_id=33)
    vels = CrazyflieFeedback(period_ms=10,)

    pose.start()
    vels.start()
    time.sleep(5)

    # run feedback for some time
    st = time.perf_counter()
    while time.perf_counter() - st < runtime:
        state = np.concatenate(
            (pose.get_pose(), vels.get_velocities())
        )
        print(f"\nState: {state}\n")
    pose.stop()
    vels.stop()


if __name__=="__main__":
    main()
