import os
import matplotlib.pyplot as plt
from PIL import Image

from deep_eikonal.core.chosen_samples import CHOSEN_SAMPLES


DATASET_PATH = 'TOSCA/raw/'
NEW_DATASET_PATH = 'geodesic_input/'


fig, axs = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Chosen samples in TOSCA', fontsize=25)
axs = axs.flatten()

for i, sample in enumerate(CHOSEN_SAMPLES):
    img_path = DATASET_PATH + sample + '.png'
    img = Image.open(img_path)
    axs[i].imshow(img)
    axs[i].axis('off')

plt.tight_layout()
plt.show()


if not os.path.exists(NEW_DATASET_PATH):
    os.mkdir(NEW_DATASET_PATH)
for file in os.listdir(NEW_DATASET_PATH):
    os.remove(NEW_DATASET_PATH + file)


def format_vertice_line(line: str) -> str:
    return line.replace('\n', ' \n').replace(' ', '\t')

def format_face_line(line: str) -> str:
    aux = lambda x : str(int(x)-1)
    verts = map(aux, line[:-1].split(' '))
    return '\t'.join(verts) + '\t\n'

for i, sample in enumerate(CHOSEN_SAMPLES):
    buffer = ''
    with open(DATASET_PATH + sample + '.vert') as file:
        lines = list(map(format_vertice_line, file.readlines()))
        nb_vertices = len(lines)
        buffer += ''.join(lines)
    with open(DATASET_PATH + sample + '.tri') as file:
        lines = list(map(format_face_line, file.readlines()))
        nb_faces = len(lines)
        buffer += ''.join(lines)
    buffer = f"{nb_vertices} {nb_faces}\n" + buffer
    with open(NEW_DATASET_PATH + f"sample_{i}.txt", 'w') as file:
        file.write(buffer)