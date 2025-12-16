# FEAST

All the scripts I utilized to get the results from the report are here, I just had to exclude the dataset and the API key for GPT-5. But if you've got one you can set up the following environment variables:

```
AZURE_OPENAI_API_KEY = '''KEY'''
AZURE_OPENAI_ENDPOINT = '''ENDPOINT'''
AZURE_OPENAI_DEPLOYMENT_NAME = '''MODEL_NAME'''
```

Then use the command:

```bash
python example_workflow_complete.py IMAGE_PAIR
```

Where the IMAGE_PAIR is the name of a pair of images that are only different at the end (the before image ends with B.jpg and the after image ends with A.jpg.


## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: Image Pair                            │
│                  (Before: B.jpg, After: A.jpg)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: GPT-5 Analysis                            │
│                                                                       │
│  python NutritionProject/gpt5.py --image_path 'B.jpg'                │
│                                                                       │
│  Output: azure_gpt4o_analysis_B.jpg.txt                              │
│  [                                                                    │
│    {"food_name": "banana", "boundary_box": [280, 25, 475, 100]},    │
│    {"food_name": "milk", "boundary_box": [25, 75, 150, 225]},       │
│    {"food_name": "muffin", "boundary_box": [200, 125, 425, 250]}    │
│  ]                                                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 2: FoodSAM Integration Setup                       │
│                                                                       │
│  foodsam = FoodSAMVLMIntegration('ckpts/sam_vit_h_4b8939.pth')      │
│  gpt5_data = load_gpt5_output_from_file('azure_gpt4o_...txt')       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 3: Parse GPT-5 Format                              │
│                                                                       │
│  parse_gpt5_output(gpt5_data)                                        │
│                                                                       │
│  Converts to internal format:                                        │
│  {                                                                    │
│    'banana': {'items': [[280, 25, 475, 100]]},                      │
│    'milk': {'items': [[25, 75, 150, 225]]},                         │
│    'muffin': {'items': [[200, 125, 425, 250]]}                      │
│  }                                                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 4: SAM Segmentation (Both Images)                  │
│                                                                       │
│  For each food item's bounding box:                                  │
│    1. Load image (before or after)                                   │
│    2. Set SAM predictor with image                                   │
│    3. Prompt SAM with bounding box                                   │
│    4. Generate binary mask                                           │
│                                                                       │
│  Result:                                                              │
│  {                                                                    │
│    'before_masks': {'banana': [mask], 'milk': [mask], ...},         │
│    'after_masks': {'banana': [mask], 'milk': [mask], ...},          │
│    'pixel_bboxes': {...}                                             │
│  }                                                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 5: Consumption Analysis                            │
│                                                                       │
│  calculator = ConsumptionCalculator()                                │
│  report = calculator.generate_full_report(results)                   │
│                                                                       │
│  For each food item:                                                 │
│    - Compare before_mask vs after_mask                               │
│    - Calculate area difference                                       │
│    - Compute consumption percentage                                  │
│                                                                       │
│  Output:                                                              │
│  {                                                                    │
│    "banana": {                                                        │
│      "items": [{                                                      │
│        "consumption_ratio": 0.75,                                    │
│        "percent_consumed": 75.0,                                     │
│        "before_area_px": 10000,                                      │
│        "after_area_px": 2500                                         │
│      }],                                                              │
│      "avg_percent_consumed": 75.0                                    │
│    },                                                                 │
│    "milk": {...},                                                     │
│    "muffin": {...}                                                    │
│  }                                                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 6: Visualization & Output                          │
│                                                                       │
│  1. Mask overlays (colored by food type)                             │
│     - before_segmented.jpg                                           │
│     - after_segmented.jpg                                            │
│                                                                       │
│  2. Bounding box visualization                                       │
│     - before_bboxes.jpg                                              │
│                                                                       │
│  3. JSON report                                                       │
│     - consumption_report.json                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Method Call Flow

```
main()
  │
  ├─→ load_gpt5_output_from_file()
  │     └─→ Parse JSON from text file
  │
  ├─→ FoodSAMVLMIntegration.__init__()
  │     └─→ Load SAM model
  │
  ├─→ process_image_pair(is_gpt5_format=True)
  │     │
  │     ├─→ Load images (cv2.imread)
  │     │
  │     ├─→ parse_gpt5_output()
  │     │     └─→ Convert to internal format
  │     │
  │     ├─→ segment_image_by_categories() [BEFORE]
  │     │     │
  │     │     └─→ For each food item:
  │     │           └─→ segment_from_bbox()
  │     │                 ├─→ predictor.set_image()
  │     │                 ├─→ predictor.predict(box=bbox)
  │     │                 └─→ Return binary mask
  │     │
  │     └─→ segment_image_by_categories() [AFTER]
  │           └─→ (same as above)
  │
  ├─→ ConsumptionCalculator.generate_full_report()
  │     │
  │     └─→ For each food name:
  │           └─→ analyze_category_consumption()
  │                 │
  │                 └─→ For each item:
  │                       └─→ compare_masks()
  │                             ├─→ calculate_mask_area()
  │                             └─→ Compute metrics
  │
  └─→ visualize_masks_overlay()
        └─→ Create colored overlays
```

## Data Structure Transformations

### 1. GPT-5 Raw Output → Parsed Format

```
INPUT (GPT-5 output file):
┌─────────────────────────────────────┐
│ ```json                              │
│ [                                    │
│   {                                  │
│     "food_name": "banana",           │
│     "boundary_box": [280, 25, ...]  │
│   }                                  │
│ ]                                    │
│ ```                                  │
└─────────────────────────────────────┘
                │
                ▼ load_gpt5_output_from_file()
                │
┌─────────────────────────────────────┐
│ [                                    │
│   {                                  │
│     "food_name": "banana",           │
│     "boundary_box": [280, 25, ...]  │
│   }                                  │
│ ]                                    │
└─────────────────────────────────────┘
```

### 2. Parsed Format → Internal Format

```
INPUT:
┌─────────────────────────────────────┐
│ [                                    │
│   {"food_name": "banana",            │
│    "boundary_box": [280, 25, ...]}  │
│ ]                                    │
└─────────────────────────────────────┘
                │
                ▼ parse_gpt5_output()
                │
OUTPUT:
┌─────────────────────────────────────┐
│ {                                    │
│   'banana': {                        │
│     'items': [[280, 25, 475, 100]]  │
│   }                                  │
│ }                                    │
└─────────────────────────────────────┘
```

### 3. Internal Format → Segmentation Results

```
INPUT (per food):
┌─────────────────────────────────────┐
│ 'banana': {                          │
│   'items': [[280, 25, 475, 100]]    │
│ }                                    │
└─────────────────────────────────────┘
                │
                ▼ segment_from_bbox() for each item
                │
OUTPUT:
┌─────────────────────────────────────┐
│ 'banana': [                          │
│   np.array([[T, F, F, ...],          │
│             [T, T, F, ...],          │
│             ...])  # Binary mask     │
│ ]                                    │
└─────────────────────────────────────┘
```

### 4. Masks → Consumption Metrics

```
INPUT:
┌─────────────────────────────────────┐
│ before_mask: 10000 True pixels       │
│ after_mask:  2500 True pixels        │
└─────────────────────────────────────┘
                │
                ▼ compare_masks()
                │
OUTPUT:
┌─────────────────────────────────────┐
│ {                                    │
│   "consumption_ratio": 0.75,         │
│   "before_area_px": 10000,           │
│   "after_area_px": 2500,             │
│   "area_diff_px": 7500,              │
│   "percent_consumed": 75.0           │
│ }                                    │
└─────────────────────────────────────┘
```

## Color Mapping System

```
Food Name → Hash → RGB Color
────────────────────────────

Predefined:
  banana     → (0, 255, 255)   [Yellow]
  milk       → (255, 255, 255) [White]
  muffin     → (165, 42, 42)   [Brown]
  strawberry → (255, 0, 127)   [Pink]
  fruit      → (0, 0, 255)     [Red]
  vegetable  → (0, 255, 0)     [Green]
  
Dynamic (unknown foods):
  hash("food") → (R, G, B)
  Deterministic: same name = same color
```

## File Organization

```
FoodSAM/
│
├── foodsam_vlm_integration.py  ← MODIFIED (main script)
│
├── NutritionProject/
│   └── gpt5.py                 ← INPUT SOURCE
│
├── results/
│   └── test_integration/
│       ├── before_segmented.jpg
│       ├── after_segmented.jpg
│       ├── before_bboxes.jpg
│       └── consumption_report.json
│
└── ckpts/
    └── sam_vit_h_4b8939.pth    ← SAM weights
```

# FoodSAM: Any Food Segmentation


This is the official PyTorch implementation of our paper:
FoodSAM: Any Food Segmentation.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/foodsam-any-food-segmentation/semantic-segmentation-on-foodseg103)](https://paperswithcode.com/sota/semantic-segmentation-on-foodseg103?p=foodsam-any-food-segmentation)

---

Segment anything Model(SAM) demonstrates significant performance on various segmentation benchmarks, showcasing its impressing zero-shot transfer capabilities on 23 diverse segmentation datasets. However, SAM lacks the class-specific information for each mask. To address the above limitation and explore the zero-shot capability of the SAM for food image segmentation, we propose a novel framework, called FoodSAM. This innovative approach integrates the coarse semantic mask with SAM-generated masks to enhance semantic
segmentation quality. Besides, it can perform instance segmentation on food images. Furthermore, FoodSAM extends its zero-shot capability to encompass panoptic segmentation by incorporating an object detector, which renders FoodSAM to effectively capture non-food object information. Remarkably, this pioneering framework stands as the first-ever work to achieve instance, panoptic, and promptable segmentation on food images. 

[[`Arxiv`](https://arxiv.org/abs/2308.05938)] 
[[`Project`]](https://starhiking.github.io/FoodSAM_Page/)
[[`IEEE TMM`]](https://ieeexplore.ieee.org/document/10306316)

![FoodSAM architecture](assets/foodsam.jpg)

FoodSAM contains three basic models: SAM, semantic segmenter, and object detector. SAM generates many class-agnostic binary masks, the semantic segmenter provides food category labels via mask-category match, and the object detector provides the non-food class for background masks. It then enhances the semantic mask via merge strategy and produces instance and panoptic results. Moreover, a seamless prompt-prior selection is integrated into the object detector to achieve promptable segmentation.

  <img src="assets/model.jpg" />

## Installation
Please follow our [installation.md](installation.md) to install.


## <a name="GettingStarted"></a>Getting Started

### Demo shell
You can run the model for semantic and panoptic segmentation in a few command lines.

#### semantic segmentation:

    # semantic segmentation for one img
    python FoodSAM/semantic.py --img_path <path/to/img> --output <path/to/output> 

    # semantic segmentation for one folder
    python FoodSAM/semantic.py --data_root <path/to/folder> --output <path/to/output>

#### panoptic segmentation:

    # panoptic segmentation for one img
    python FoodSAM/panoptic.py --img_path <path/to/img> --output <path/to/output>

    # panoptic segmentation for one folder
    python FoodSAM/panoptic.py --data_root <path/to/folder> --output <path/to/output>



### Evaluation shell
Furthermore, by setting `args.eval` to true, the model can output the semantic masks and evaluate the metrics. 
Here are examples of semantic segmentation and panoptic segmentation on the FoodSeg103 dataset:
```
python FoodSAM/semantic.py --data_root dataset/FoodSeg103/Images --output Output/Semantic_Results --eval 
```
```
python FoodSAM/panoptic.py --data_root dataset/FoodSeg103/Images --output Output/Panoptic_Results
```

## Quantitative results

### FoodSeg103
| Method | mIou | aAcc | mAcc 
| :-: | :- | -: | :-: |  
|[SETR_MLA(baseline)](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1) | 45.10 | 83.53 | 57.44
FoodSAM | 46.42 | 84.10 |  58.27

### UECFOODPIXCOMPLETE

| Method | mIou | aAcc | mAcc 
| :-: | :- | -: | :-: |  
|[deeplabV3+ (baseline)](https://github.com/HitBadTrap/Foodseg-uecfoodpix)| 65.61 |88.20| 77.56
FoodSAM | 66.14 |88.47 |78.01

## Qualitative results

### cross domain results

 <img src="assets/crossdomain.png">

### semantic segmentation results 

 <img src="assets/semantic.jpg">
 
---

 <img src="assets/semantic_compare.jpg">
 
### instance segmentation results
<img src="assets/instance_compare.jpg">

### panoptic segmentation results
<img src="assets/panoptic_compare.jpg">

### promptable segmentation results
<img src="assets/prompt_vis.jpg">

## Acknowledgements

A large part of the code is borrowed from the following wonderful works:

1. [Segmentation Anything](https://github.com/facebookresearch/segment-anything)

2. [UniDet](https://github.com/xingyizhou/UniDet)

3. [FoodSeg103](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1)

4. [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Citation
If you want to cite our work, please use this:

```
@ARTICLE{10306316,
  author={Lan, Xing and Lyu, Jiayi and Jiang, Hanyu and Dong, Kun and Niu, Zehai and Zhang, Yi and Xue, Jian},
  journal={IEEE Transactions on Multimedia}, 
  title={FoodSAM: Any Food Segmentation}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TMM.2023.3330047}
}

@misc{lan2023foodsam,
      title={FoodSAM: Any Food Segmentation}, 
      author={Xing Lan and Jiayi Lyu and Hanyu Jiang and Kun Dong and Zehai Niu and Yi Zhang and Jian Xue},
      year={2023},
      eprint={2308.05938},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
