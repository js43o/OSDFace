# edge_dists_demo.py
import torch
import torch.nn as nn
import pyiqa


class EdgeDetectionModel(nn.Module):
    def __init__(self):
        super(EdgeDetectionModel, self).__init__()
        # Sobel filters for x and y directions
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # Define Sobel kernels
        sobel_x_kernel = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        sobel_y_kernel = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        )

        # Load kernels into convolution layers
        self.sobel_x.weight = nn.Parameter(sobel_x_kernel.view(1, 1, 3, 3))
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel.view(1, 1, 3, 3))

        # Freeze Sobel filter weights
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        # If input is [B, 3, H, W], convert to grayscale by averaging channels
        if x.dim() == 4 and x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)  # now [B, 1, H, W]
        # Apply Sobel filters
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        # Compute gradient magnitude
        edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return edges


# EA-DISTS 손실 함수 클래스
class EdgeAwareDISTSLoss:
    def __init__(self, device):
        self.dists_loss = pyiqa.create_metric("dists", device=device, as_loss=True)
        self.edge_model = EdgeDetectionModel().to(device)
        self.edge_model.requires_grad_(False)

    def __call__(self, pred, gt):
        # pred.shape == gt.shape == [B, 3, 512, 512], 값 범위 [0, 1]
        pred_norm = (pred + 1) / 2
        gt_norm = (gt + 1) / 2

        d_loss = self.dists_loss(pred_norm, gt_norm)

        edge_pred = self.edge_model(pred_norm)
        edge_gt = self.edge_model(gt_norm)
        e_loss = self.dists_loss(edge_pred, edge_gt)

        total_loss = d_loss + e_loss

        return total_loss


def main():
    # Choose device: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create the dists metric as a loss function on the selected device
    dists_loss = pyiqa.create_metric("dists", device=device, as_loss=True)

    # Instantiate and freeze the edge detection model
    edge_model = EdgeDetectionModel().to(device)
    edge_model.requires_grad_(False)

    # Prepare fake data: batch size 2, 3 channels, 512x512
    pred_image = torch.rand(2, 3, 512, 512, device=device)
    gt_image = torch.rand(2, 3, 512, 512, device=device)

    # Compute losses
    loss_dict = {}

    # Compute dists loss on original images
    d_loss = dists_loss(pred_image, gt_image)
    loss_dict["loss_dists"] = d_loss

    # Compute dists loss on edge maps
    edge_pred = edge_model(pred_image)
    edge_gt = edge_model(gt_image)
    e_loss = dists_loss(edge_pred, edge_gt)
    loss_dict["loss_edge"] = e_loss

    # Total combined loss
    total_loss = d_loss + e_loss
    loss_dict["loss_total"] = total_loss

    # Print all losses
    print("All losses:", {k: v.item() for k, v in loss_dict.items()})


if __name__ == "__main__":
    main()
