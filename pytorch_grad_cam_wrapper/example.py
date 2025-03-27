from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam_wrapper.Wrapper import generate_gradcam
from einops import rearrange
import cv2

batch["rgb"] = rearrange(batch["rgb"], 'b t v c h w -> (b t v) c h w')
grayscale_cam = generate_gradcam(batch, "rgb", "action", self.net, self.net.net.vision_tower.convs[0].proj, RawScoresOutputTarget(), GradCAM)
# grayscale_cam = generate_gradcam(batch, "coord", "action", self.net, self.net.position_embedding_3d.layers[0], RawScoresOutputTarget(), GradCAM)
visualization = show_cam_on_image(rearrange(batch['rgb'][0], 'c h w -> h w c').cpu().numpy(), grayscale_cam[0], use_rgb=True)
bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
cv2.imwrite('gradcam.jpg', bgr)