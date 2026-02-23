import numpy as np
import torch
from .retargeter import Retargeter
from .utils import retargeter_utils


class AbsoluteRetargeter(Retargeter):
    """Retargeter using absolute fingertip position matching instead of key vectors."""

    def optimize_orcahand_joint_angles(self, manohand_joint_pos, opt_steps=2):
        manohand_joint_pos = torch.from_numpy(manohand_joint_pos).to(self.device)
        manohand_fingertips, _ = retargeter_utils.extract_mano_fingertips_and_palm(
            manohand_joint_pos, self.fingers, self.source)

        for _ in range(opt_steps):
            urdfhand_joint_angles = torch.zeros(self.chain.n_joints, device=self.device)
            urdfhand_joint_angles[self.joint_reorder_indices] = self.orcahand_joint_angles / (180.0 / np.pi)
            urdfhand_fingertips, _ = retargeter_utils.extract_orca_fingertips_and_palm(
                self.chain, urdfhand_joint_angles, self.optimization_frames,
                self.hand_type, self.fingers, self.root,
                fingertip_offsets=self._fingertip_offsets)

            loss = sum(
                self.loss_coeffs[i] * torch.norm(manohand_fingertips[finger] - urdfhand_fingertips[finger]) ** 2
                for i, finger in enumerate(self.fingers)
            )
            loss += torch.sum(self.regularizer_weights * (self.orcahand_joint_angles - self.regularizer_zeros) ** 2)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            with torch.no_grad():
                self.orcahand_joint_angles.clamp_(self.joint_angle_limits_lower, self.joint_angle_limits_upper)

        return self.orcahand_joint_angles.detach().cpu().numpy()
