# diag2_debug.py
import numpy as np, os
DATA = r"C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface"
for rel in [
    '16--Award_Ceremony/16_Award_Ceremony_Awards_Ceremony_16_84',
    '44--Aerobics/44_Aerobics_Aerobics_44_173',
    '57--Angler/57_Angler_peoplefishing_57_250',
]:
    event, stem = rel.split('/')
    npz = os.path.join(DATA, 'kd_cache/val', event, stem + '.npz')
    print(npz, 'exists:', os.path.isfile(npz))
    if os.path.isfile(npz):
        z = np.load(npz)
        print('  keys:', list(z.keys()))
        print('  boxes shape:', z['boxes'].shape)
        print('  boxes:', z['boxes'].reshape(-1, 4).tolist())