# vaik-text-recognition-pb-inference

Inference by text recognition PB model


## Install

``` shell
pip install git+https://github.com/vaik-info/vaik-text-recognition-pb-inference.git
```

## Usage

### Example
```python
import numpy as np
from PIL import Image
from vaik_text_recognition_pb_inference.pb_model import PbModel

classes = PbModel.char_json_read('/home/kentaro/Github/vaik-text-recognition-pb-trainer/data/jpn_character.json')
model_path = '/home/kentaro/.vaik_text_recognition_pb_trainer/output_model/2023-02-04-20-58-00/step-5000_batch-16_epoch-24_loss_0.3250_val_loss_0.1010'
model = PbModel(model_path, classes)

image1 = np.asarray(Image.open("/home/kentaro/Desktop/images/いわき_0333.png").convert('RGB'))

output, raw_pred = model.inference([image1, image1, image1], batch_size=2)
```


#### Output

- output

```text
[
  {
    'text': [
      'い山形',
      'いわ形',
      'い九形',
      'い小形',
      'い山き',
      'い州形',
      'いわき',
      'い九き',
      'い山保',
      'い形'
    ],
    'classes': [
      [
        0,
        114,
        1331,
        969,
        0
      ],
      [
        0,
        114,
        156,
        969,
        0
      ],
      [
        0,
        114,
        911,
        969,
        0
      ],
      [
        0,
        114,
        1608,
        969,
        0
      ],
      [
        0,
        114,
        1331,
        119,
        0
      ],
      [
        0,
        114,
        1503,
        969,
        0
      ],
      [
        0,
        114,
        156,
        119,
        0
      ],
      [
        0,
        114,
        911,
        119,
        0
      ],
      [
        0,
        114,
        1331,
        2783,
        0
      ],
      [
        0,
        114,
        969,
        0
      ]
    ],
    'scores': [
      0.010506187565624714,
      0.008148221299052238,
      0.008104778826236725,
      0.003845307743176818,
      0.003661569906398654,
      0.003305705962702632,
      0.002839782042428851,
      0.002824641764163971,
      0.002469017868861556,
      0.002012777840718627
    ]
  },
  ・・・
    ]
  }
]
```