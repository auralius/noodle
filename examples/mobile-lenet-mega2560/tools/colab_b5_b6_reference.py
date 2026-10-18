# Run this cell in the existing Case 2 Colab validation notebook,
# AFTER Section 7.4 has defined NV_KERAS_MODEL, NV_FEATURES, NV_GT and NV_IMG16.
# Does not train, export, or change any model parameters.
import numpy as np
import pandas as pd
import tensorflow as tf
import zlib
from pathlib import Path

PROBE_INDICES = [30, 42, 80]
for layer_name in ('B5_pw_relu', 'B6_pw_relu'):
    assert NV_KERAS_MODEL.get_layer(layer_name) is not None
submodel = tf.keras.Model(
    inputs=NV_KERAS_MODEL.inputs,
    outputs=[NV_KERAS_MODEL.get_layer('B5_pw_relu').output,
             NV_KERAS_MODEL.get_layer('B6_pw_relu').output],
)
b5, b6 = submodel(NV_FEATURES[PROBE_INDICES].astype(np.float32), training=False)
b5 = np.asarray(b5, dtype=np.float32)
b6 = np.asarray(b6, dtype=np.float32)
assert b5.shape == (3, 4, 4, 24), b5.shape
assert b6.shape == (3, 4, 4, 24), b6.shape
b5_mean = b5.mean(axis=(1,2))
b6_mean = b6.mean(axis=(1,2))
rows = []
for j, idx in enumerate(PROBE_INDICES):
    entry = {'idx':idx, 'gt':int(NV_GT[idx]),
             'pixel_crc32':f'{zlib.crc32(NV_IMG16[idx].tobytes()):08X}'}
    for c in range(24):
        entry[f'b5_mean{c}'] = float(b5_mean[j,c])
        entry[f'b6_mean{c}'] = float(b6_mean[j,c])
    rows.append(entry)
    print('Index',idx,'GT',entry['gt'],
          'B5 mean preview',np.round(b5_mean[j,:4],5),
          'B6 mean preview',np.round(b6_mean[j,:4],5))
output = Path('/content/case2_fp32_keras_b5_b6_reference.csv')
pd.DataFrame(rows).to_csv(output, index=False, float_format='%.9g')
print('Saved',output)
# Optional download:
# from google.colab import files
# files.download(str(output))
