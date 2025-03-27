import torch
import torch.nn as nn

class NetWrapper(nn.Module):
    def __init__(self, net, x, y):
        """
        Initialize the NetWrapper module.

        Args:
        - net: The original neural network model, which takes a dictionary `batch` as input and outputs a dictionary `pred`.
        - x: String, representing the key name of the primary input tensor (e.g., 'image').
        - y: String, representing the key name of the primary output tensor (e.g., 'a').
        """
        super().__init__()
        self.net = net
        self.x_key = x
        self.y_key = y
        self._other_inputs = {}  # Used to store other tensors in the batch except for batch[x]

    def set_other_inputs(self, other_inputs):
        """
        Set other inputs in the batch except for batch[x].

        Args:
        - other_inputs: Dictionary containing key-value pairs in the batch except for batch[x].
        """
        self._other_inputs = other_inputs

    def forward(self, input_tensor):
        """
        Forward pass.

        Args:
        - input_tensor: Tensor, corresponding to batch[x].

        Returns:
        - The output of the net (a dictionary `pred`).
        """
        # Construct the complete input dictionary
        input_dict = {self.x_key: input_tensor}
        input_dict.update(self._other_inputs)
        # Call the original model and return the output
        pred, _ = self.net(input_dict)
        return [pred[self.y_key].sum()]

def generate_gradcam(batch, input_key, output_key, net, target_layer, target, cam_method):
    """
    Generate CAM visualization using the provided batch data, network, target layer, target, and specified CAM algorithm.

    Args:
        batch (dict): A dictionary containing the model's input data, e.g., {'image': tensor, 'mask': tensor}.
        input_key (str): The key in the batch corresponding to the primary input tensor (e.g., 'image').
        output_key (str): The key in the pred corresponding to the primary output tensor (e.g., 'action').
        net (nn.Module): The original neural network model, which accepts a dictionary as input and returns a dictionary as output.
        target_layer (nn.Module): The target layer used for CAM computation.
        target (object): The target for CAM, such as an instance of RawScoresOutputTarget or a custom target.
        cam_method (class): The CAM algorithm class to use, e.g., GradCAM, HiResCAM, ScoreCAM, etc. Defaults to GradCAM.

    Returns:
        numpy.ndarray: The generated grayscale CAM visualization (grayscale_cam).
    """
    # Step 1: Instantiate NetWrapper to adapt the network for CAM computation
    wrapped_net = NetWrapper(net=net, x=input_key, y=output_key)
    
    # Step 2: Extract other inputs from the batch except for the input_key
    other_inputs = {key: value for key, value in batch.items() if key != input_key}
    
    # Step 3: Set other inputs in the NetWrapper
    wrapped_net.set_other_inputs(other_inputs)
    
    # Step 4: Initialize the CAM object using the specified CAM algorithm class
    with cam_method(model=wrapped_net, target_layers=[target_layer]) as cam:
        # Step 5: Extract the primary input tensor from the batch
        input_tensor = batch[input_key]
        
        # Step 6: Generate the grayscale CAM visualization
        grayscale_cam = cam(input_tensor=input_tensor, targets=[target])
    
    return grayscale_cam