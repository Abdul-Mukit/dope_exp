import cv2
import matplotlib.pyplot as plt
import torch
from detector import *

## Settings
name = 'mustard'
# net_path = 'data/net/mustard_60.pth'
net_path = 'data/net/cautery_s2_60.pth'
gpu_id = 0
img_path = 'data/images/cautery_DR_2.png'
# img_path = 'data/images/cautery_real_1.jpg'



# Function for visualizing feature maps
def viz_layer(layer, n_filters=9):
    fig = plt.figure(figsize=(20, 20))
    row = 1
    for i in range(n_filters):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i + 1))

# load color image
in_img = cv2.imread(img_path)
# in_img = cv2.resize(in_img, (640, 480))
in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
# plot image
plt.imshow(in_img)


model = ModelData(name, net_path, gpu_id)
model.load_net_model()
net_model = model.net

# Run network inference
image_tensor = transform(in_img)
image_torch = Variable(image_tensor).cuda().unsqueeze(0)
out, seg = net_model(image_torch)
vertex2 = out[-1][0].cpu()
aff = seg[-1][0].cpu()

# View the vertex and affinities
viz_layer(vertex2)
viz_layer(aff, n_filters=16)

plt.show()

