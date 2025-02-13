# PnP Node
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>

from scipy.spatial import transform
import torch
from kornia.geometry import conversions 
import open3d
import numpy as np
import random
from scipy.spatial.transform import Rotation
import copy

from ddn.pytorch.node import *
import ddn.pytorch.geometry_utilities as geo

class PointCloudRegistration(AbstractDeclarativeNode):
    """"""
    def __init__(self,
        eps=1e-8,
        gamma=None,
        objective_type='reproj',
        alpha=1.0,
        chunk_size=None,
        ransac_max_num_iterations=1000,
        ransac_threshold=0.1
        ):
        super().__init__(eps=eps, gamma=gamma, chunk_size=chunk_size)
        self.objective_type = objective_type
        self.alpha = alpha
        self.ransac_max_num_iterations = ransac_max_num_iterations
        self.ransac_threshold = ransac_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def objective(self, cloud_src, cloud_tgt, w, y):
        """Weighted registration error

        Arguments:
            cloud_src: (b, n, 3) Torch tensor,
                batch of source point cloud

            cloud_tgt: (b, n, 3) Torch tensor,
                batch of 3D target point cloud

            w: (b, n) Torch tensor,
                batch of weight vectors

            T: (b, 6) Torch tensor,
                batch of transformation parameters
                format:
                    T[:, 0:3]: angle-axis rotation vector
                    T[:, 3:6]: translation vector

        Return Values:
            objective value: (b, ) Torch tensor
        """
        if self.objective_type is 'cosine':
            return self.objective_cosine(cloud_src, cloud_tgt, w, y)
        elif self.objective_type is 'reproj':
            return self.objective_reproj(cloud_src, cloud_tgt, w, y)
        elif self.objective_type is 'reproj_huber':
            return self.objective_reproj_huber(cloud_src, cloud_tgt, w, y)

    def objective_cosine(self, p2d, p3d, w, K, theta):
        """Weighted cosine distance error
        f(p2d, p3d, w, y) = sum_{i=1}^n
            w_i * (1 - p2d_i^T N(R(y) p3d_i + t(y)))
            where N(p) = p / ||p||
        """
        p2d_bearings = geo.points_to_bearings(p2d, K)
        p3d_transform = geo.transform_and_normalise_points_by_theta(p3d, theta)
        return torch.einsum('bn,bn->b', (w, 1.0 - torch.einsum('bnd,bnd->bn',
            (p2d_bearings, p3d_transform))))

    def objective_reproj(self, cloud_src, cloud_tgt, w, T):
        """Weighted squared reprojection error
        f(cloud_src, cloud_tgt, w, T) = sum_{i=1}^n
            w_i * ||pi(cloud_src, T) - cloud_tgt||_2^2
            where pi(cloud_src, T) = R(T) * cloud_src + t(T)
        """
        cloud_src_transformed = geo.transform_points_by_theta(cloud_src, T)
        z2 = torch.sum((cloud_src_transformed - cloud_tgt) ** 2, dim=-1)
        return torch.einsum('bn,bn->b', (w, z2))

    def objective_reproj_huber(self, cloud_src, cloud_tgt, w, T):
        """Weighted Huber reprojection error
        f(p2d, p3d, w, K, y) = sum_{i=1}^n
            w_i * rho(pi(p3d_i, K, y) - p2d_i, alpha)
            where rho(z, alpha) = / 0.5 z^2 for |z| <= alpha
                                  \ alpha * (|z| - 0.5 * alpha) else
            and pi(p, K, y) = h2i(K * (R(y) * p + t(y)))
            where h2i(x) = [x1 / x3, x2 / x3]
        """
        def huber(z2, alpha=1.0):
            return torch.where(z2 <= alpha ** 2, 0.5 * z2, alpha * (
                z2.sqrt() - 0.5 * alpha))
        cloud_src_transformed = geo.transform_points_by_theta(cloud_src, T)
        z2 = torch.sum((cloud_src_transformed - cloud_tgt) ** 2, dim=-1)
        return torch.einsum('bn,bn->b', (w, huber(z2, self.alpha)))

    def solve(self, cloud_src, cloud_tgt, w):
        cloud_src = cloud_src.detach()
        cloud_tgt = cloud_tgt.detach()
        w = w.detach()
        T = self._initialise_transformation(cloud_src, cloud_tgt, w).requires_grad_()
        T = self._run_optimisation(cloud_src, cloud_tgt, w, y=T)
        # # Alternatively, disentangle batch element optimisation:
        # for i in range(p2d.size(0)):
        #     Ki = K[i:(i+1),...] if K is not None else None
        #     theta[i, :] = self._run_optimisation(p2d[i:(i+1),...],
        #         p3d[i:(i+1),...], w[i:(i+1),...], Ki, y=theta[i:(i+1),...])
        return T.detach(), None

    def _initialise_transformation(self, cloud_src, cloud_tgt, w):
        return self._ransac_solve_transformation(cloud_src, cloud_tgt,
            self.ransac_max_num_iterations, self.ransac_threshold)

    def _ransac_solve_transformation(self, cloud_src, cloud_tgt, max_num_iterations, reprojection_error_threshold):
        T = cloud_src.new_zeros(cloud_src.size(0), 6)
        b, N, _ = cloud_src.shape
        # cloud_src_np = cloud_src.cpu().numpy()
        # cloud_tgt_np = cloud_tgt.cpu().numpy()

        # # calculate centroid of cloud_src and cloud_tgt
        # cloud_src_centroid = np.average(cloud_src_np, axis = 1)
        # cloud_tgt_centroid = np.average(cloud_tgt_np, axis = 1)
        # assert cloud_src_centroid.shape[1] == 3
        # assert cloud_tgt_centroid.shape[1] == 3

        # # centralize point clouds to zero
        # cloud_src_centroid_repeat = np.repeat(cloud_src_centroid[:, np.newaxis, :], N, axis = 1)
        # cloud_tgt_centroid_repeat = np.repeat(cloud_tgt_centroid[:, np.newaxis, :], N, axis = 1)
        # cloud_src_demean = cloud_src_np - cloud_src_centroid_repeat
        # cloud_tgt_demean = cloud_tgt_np - cloud_tgt_centroid_repeat
        
        for b_i in range(b):
            for _ in range(max_num_iterations):
                random_idx = random.sample(range(N), 3)
                cloud_src_i = cloud_src[b_i, random_idx, :]
                cloud_tgt_i = cloud_tgt[b_i, random_idx, :]
                
                T_i = self._solve_transformation(cloud_src_i, cloud_tgt_i)
                n_inliers_i = self._calculate_number_of_inliers(cloud_src[b_i], cloud_tgt[b_i], T_i, reprojection_error_threshold)
                print(n_inliers_i / N)
                if n_inliers_i > N * 0.80:
                    break
            T[b_i, :] = T_i.double()
        return T

    def _solve_transformation(self, cloud_src, cloud_tgt):
        T = torch.zeros(1, 6).to(self.device)
        TE = open3d.t.pipelines.registration.TransformationEstimationPointToPoint()

        corr = torch.zeros(3, 1, dtype = torch.int64).cuda(0)
        corr[:, 0] = torch.arange(0, 3)
        cloud_src_cloud = open3d.t.geometry.PointCloud(device=open3d.core.Device("CUDA:0"))
        cloud_tgt_cloud = open3d.t.geometry.PointCloud(device=open3d.core.Device("CUDA:0"))
        cloud_src_cloud.point["positions"] = open3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(cloud_src))
        cloud_tgt_cloud.point["positions"] = open3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(cloud_tgt))
        corr_o3d_tensor = open3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(corr))

        transformation_o3d = TE.compute_transformation(cloud_src_cloud, cloud_tgt_cloud, corr_o3d_tensor)
        # calculate optimal translation vector
        t = transformation_o3d[:3, 3]
        R_axis_angle = conversions.rotation_matrix_to_angle_axis(torch.utils.dlpack.from_dlpack(transformation_o3d[:3, :3].to_dlpack()).contiguous())
        T[0, :3] = R_axis_angle
        T[0, 3:] = torch.utils.dlpack.from_dlpack(t.to_dlpack())

        return T

    def _calculate_number_of_inliers(self, cloud_src, cloud_tgt, T, transformation_error_threshold):
        # rot_mat = Rotation.from_rotvec(T[0, :3]).as_matrix()
        tf_mat = torch.zeros(4, 4).to(self.device)
        q = conversions.angle_axis_to_quaternion(T[0, :3].view(1, 3))

        tf_mat[:3, :3] = conversions.quaternion_to_rotation_matrix(q)
        tf_mat[:3, 3] = T[0, 3:]
        tf_mat[3, 3] = 1
    
        # cloud_src_transformed = geo.transform_points_by_theta(torch.from_numpy(cloud_src).unsqueeze(0).double(), torch.from_numpy(T).double())
        cloud_src_cloud = open3d.t.geometry.PointCloud(device=open3d.core.Device("CUDA:0"))
        cloud_src_cloud.point["positions"] = open3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(cloud_src))

        # transform src point cloud
        tf_mat_o3d_tensor = open3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tf_mat))
        cloud_src_transformed = cloud_src_cloud.clone().cuda(0).transform(tf_mat_o3d_tensor)
        # calculate number of correspondences that is within the threshold
        transformation_error = torch.norm((torch.utils.dlpack.from_dlpack(cloud_src_transformed.point["positions"].to_dlpack()) - cloud_tgt), dim = 1)
        print(transformation_error)
        # print(transformation_error)
        number_of_inliers = torch.sum(transformation_error < transformation_error_threshold)
        print(transformation_error < transformation_error_threshold)
        print(number_of_inliers)
        # print(number_of_inliers)
        return number_of_inliers

    def _run_optimisation(self, *xs, y):
        with torch.enable_grad():
            opt = torch.optim.LBFGS([y],
                                    lr=1.0,
                                    max_iter=1000,
                                    max_eval=None,
                                    tolerance_grad=1e-40,
                                    tolerance_change=1e-40,
                                    history_size=100,
                                    line_search_fn="strong_wolfe"
                                    )
            def reevaluate():
                opt.zero_grad()
                f = self.objective(*xs, y=y).sum() # sum over batch elements
                f.backward()
                return f
            opt.step(reevaluate)
        return y