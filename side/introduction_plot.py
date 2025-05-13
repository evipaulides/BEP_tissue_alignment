import matplotlib.pyplot as plt
from PIL import Image
import os

def plot_images(image_paths, subtitles, figsize=(16, 8)):
    """
    Plot 8 images (2 rows × 4 cols) with custom subtitles for each image.

    :param image_paths: List of 8 image file paths in order (row-wise)
    :param subtitles: List of 8 subtitles (one for each image)
    :param figsize: Figure size
    """
    assert len(image_paths) == 8 and len(subtitles) == 8, "Provide exactly 8 images and 8 subtitles."

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.subplots_adjust(hspace=0.4)

    for idx, (img_path, subtitle) in enumerate(zip(image_paths, subtitles)):
        row, col = divmod(idx, 4)
        ax = axes[row][col]
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(subtitle, fontsize=20)
        ax.axis('off')

    # Add title
    fig.suptitle('Comparison of H&E and IHC Stained Slides Across Two Patients', fontsize=30, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
    plt.tight_layout()
    plt.savefig('multistain_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
image_paths = [
    'data/Multi-stain_images/patient_1/HE.tif',
    'data/Multi-stain_images/patient_1/BCAT.tif',
    'data/Multi-stain_images/patient_1/KI-MELANA.tif',
    'data/Multi-stain_images/patient_1/PRAME.tif',
    'data/Multi-stain_images/patient_2/HE.tif',
    'data/Multi-stain_images/patient_2/BAP1.tif',
    'data/Multi-stain_images/patient_2/MELANA.tif',
    'data/Multi-stain_images/patient_2/PRAME.tif'
]

subtitles = [
    'H&E',
    'IHC (BCAT)',
    'IHC (KI-MELANA)',
    'IHC (PRAME)',
    'H&E',
    'IHC (BAP1)',
    'IHC (MELANA)',
    'IHC (PRAME)'
]

plot_images(image_paths, subtitles)
