import matplotlib.pyplot as plt
from dataloader import valid_loader
import matplotlib.pyplot as plt

images, masks = next(iter(valid_loader))  # get one batch

for i in range(2):  # display first 2 samples
    img = images[i].permute(1, 2, 0).numpy()  # convert to HWC for plotting
    mask = masks[i].numpy()

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()
    