**[When Negation Is a Geometry Problem in Vision-Language Models](https://arxiv.org/pdf/2603.20554)**, CVPR 2026 (Multimodal Algorithmic Reasoning Workshop)

- Paper: https://arxiv.org/pdf/2603.20554

## Code structure

- `steer.py`: main script (trains steering directions from `data/train_data.json`, retrieves images from COCO `val2014`, and evaluates with a VQA model).
- `data/train_data.json`: positive→negative sentence pairs used to train the layerwise steering directions.
- `data/simpleneg.json`: evaluation benchmark.

## Setup

### 1) Install Python dependencies

Create a new PyTorch environment and install deps:

```bash
pip install git+https://github.com/openai/CLIP.git
pip install open_clip_torch transformers accelerate pillow numpy matplotlib scikit-learn
```

### 2) Download baseline checkpoints

Download the [baseline models](https://drive.google.com/drive/folders/17TtRZLnvIZUK7vvXjT32zXVIkJczi2OI?usp=sharing). You can also download them yourself from the original repos.  


### 3) Download COCO 2014 val images

Please download the [MS-COCO 2014 validation images](http://images.cocodataset.org/zips/val2014.zip) in:

```text
datasets/val2014/
	COCO_val2014_000000000042.jpg
	...
```

## Run

From the repository root:

```bash
python steer.py
```

Common options:

- Change CLIP backbone:

```bash
python steer.py --model_name "ViT-L/14"
```

- Set any image database size you want (e.g., increase to 100,000):

```bash
python steer.py --img_database_size 100000
```

- Skip NegBench/NegCLIP baselines (OpenCLIP checkpoints) entirely:

```bash
python steer.py --openclip_model_name none
```

- Use any Qwen-based VLM-as-a-judge:

```bash
python steer.py --vqa_model_id "Qwen/Qwen3-VL-4B-Instruct"
```

## Outputs

You’ll typically see:

- `baseline_results.json`
- `conclip_results.json`
- `steering_results.json`
- (if OpenCLIP baselines enabled) `negbench_results.json`, `negclip_results.json`
