# TEST REGISTRATION DEEP DECLARATIVE NODES
#
# Zherui Zhang <zherui.bill.zhang@gmail.com>
#
# When running from the command-line make sure that the "ddn" package has been added to the PYTHONPATH:
#   $ export PYTHONPATH=${PYTHONPATH}: ../ddn
#   $ python testPyTorchDeclNodes.py

import torch
from torch.autograd import grad
from torch.autograd import gradcheck
from scipy.spatial.transform import Rotation

import sys

sys.path.append("../")
from ddn.pytorch.registration_node import *

# Generate point correspondences
b = 4
N = 12
torch.manual_seed(0)
cloud_src_test = torch.randn(b, N, 3, dtype=torch.double)
R_test = torch.tensor(
    [[0.8660254, -0.5000000,  0.0000000],
     [0.5000000, 0.8660254, 0.0000000],
     [0.0000000, 0.0000000, 1.0000000]], dtype=torch.double)
R_test_batch = R_test.unsqueeze(0).repeat(b, 1, 1)
t_test = torch.randn(b, 3)
cloud_tgt_test = torch.einsum('brs,bms->bmr', (R_test_batch, cloud_src_test)) + t_test.unsqueeze(1)

cloud_tgt_verify = R_test_batch[0, :, :] @ cloud_src_test[0, :, :].T

cloud_tgt_test_noisy = cloud_tgt_test + 0.04 * torch.randn(b, N, 3, dtype = torch.double) # add noise
# p2d[:, 0:1, :] = torch.randn(b, 1, 2, dtype=torch.double) # add outliers

# Plot:
# import matplotlib.pyplot as plt
# p2d_np = p2d.cpu().numpy()
# p3d_proj_np = geo.project_points_by_theta(p3d, theta).cpu().numpy()
# plt.scatter(p2d_np[0, :, 0], p2d_np[0, :, 1], s=10, c='k', alpha=1.0, marker='s')
# plt.scatter(p3d_proj_np[0, :, 0], p3d_proj_np[0, :, 1], s=10, c='r', alpha=1.0, marker='o')
# plt.show()

w = torch.ones(b, N, dtype=torch.double) # bxn
w = w.abs() # Weights must be positive and sum to 1 per batch element
w = w.div(w.sum(-1).unsqueeze(-1))

# Create a PnP problem and create a declarative layer:
# node = PnP(objective_type='cosine')
node = PointCloudRegistration(objective_type='reproj', chunk_size=None)
# node = PnP(objective_type='reproj_huber', alpha=0.1)
DL = DeclarativeLayer(node)

cloud_src_test = cloud_src_test.requires_grad_()
cloud_tgt_test_noisy = cloud_tgt_test_noisy.requires_grad_()
w = w.requires_grad_()

# DL, p2d, p3d, w, K = DL.cuda(0), p2d.cuda(0), p3d.cuda(0), w.cuda(0), K.cuda(0) if K is not None else None # Move everything to GPU

# Run forward pass:
y = DL(cloud_src_test, cloud_tgt_test_noisy, w)

# Compute objective function value:
f = node.objective(cloud_src_test, cloud_tgt_test_noisy, w, y=y)

# Compute gradient:
Dy = grad(y, (cloud_src_test, cloud_tgt_test_noisy, w), grad_outputs=torch.ones_like(y))
y_gt = np.zeros((b, 6))

y_gt[:, :3] = Rotation.from_matrix(R_test).as_rotvec()
y_gt[:, 3 : 6] = t_test

# print("Input p2d:\n{}".format(p2d.detach().cpu().numpy()))
# print("Input p3d:\n{}".format(p3d.detach().cpu().numpy()))
# print("Input w:\n{}".format(w.detach().cpu().numpy()))
# print("Input K:\n{}".format(K))
print("Theta Ground-Truth:\n{}".format(y_gt))
print("Theta Estimated:\n{}".format(y.detach().cpu().numpy()))
print("Objective Function Value:\n{}".format(f.detach().cpu().numpy()))
# print("Dy:\n{}\n{}\n{}".format(Dy[0].detach().cpu().numpy(), Dy[1].detach().cpu().numpy(), Dy[2].detach().cpu().numpy()))

# Run gradcheck:
# DL, p2d, p3d, w, K = DL.cpu(), p2d.cpu(), p3d.cpu(), w.cpu(), K.cpu() if K is not None else None # Move everything to CPU
test = gradcheck(DL, (cloud_src_test, cloud_tgt_test_noisy, w), eps=1e-4, atol=1e-4, rtol=1e-4, raise_exception=True)
print("gradcheck passed:", test)