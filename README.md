# SF-Swin-UNETR: COVID-19 CT Infection Segmentation

Swin-UNETR with a Spatial-Frequency Gate (SF-Gate) for COVID-19 lung infection
segmentation on the COVID-19-CT-Seg benchmark.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

On a machine where GPU PyTorch is already installed (e.g. Colab / RunPod), do
not reinstall torch; install only the missing packages individually (see the
note in `requirements.txt`).

### 2. Prepare data

Place the Zenodo COVID-19-CT-Seg dataset as follows:

```
data/zenodo/
└── COVID-19-CT-Seg/
    ├── COVID-19-CT-Seg_20cases/  # CT volumes
    └── Infection_Mask/            # binary infection masks
```

### 3. Verify data

```bash
python verify_data.py --data_root ./data/zenodo
```

### 4. Train

```bash
python train.py \
    --data_root ./data/zenodo \
    --spatial_size 96 96 96 \
    --feature_size 48 \
    --batch_size 4 \
    --epochs 500 \
    --val_interval 10 \
    --exp_name covid_sf_swin
```

Run the baseline-vs-SF-Gate comparison:

```bash
python run_experiments.py --data_root ./data/zenodo --epochs 500
```

## How to run the full pipeline

Run the scripts in this order:

1. `python verify_data.py --data_root ./data/zenodo` — check the dataset loads.
2. `python quickstart.py --data_root ./data/zenodo` — end-to-end sanity check
   (data sanity, dataloaders, overfit) before full training.
3. `python train.py --data_root ./data/zenodo ...` — train one model.
4. `python run_experiments.py --data_root ./data/zenodo --epochs 500` — run the
   baseline-vs-SF-Gate comparison (trains both models back to back).
5. `python eval_metrics.py --checkpoint ./outputs/sf_swin_unetr/best_model.pth` —
   compute Dice + HD95 on the validation split (no retraining needed).
6. `python inference.py ...` then `python visualize.py ...` — generate and plot
   qualitative results on the validation split.

Every script supports `--help`. Auxiliary scripts: `smoke_test.py` (zero-data
smoke test) and `data_stats.py` (dataset statistics).

## Project structure

```
proj_code/
├── configs/config.py        # training configuration
├── data/dataset.py          # data loading + transforms
├── models/sf_gate.py        # SF-Gate module
├── models/sf_swin_unetr.py  # main model
├── train.py                 # training script
├── run_experiments.py       # baseline vs. SF-Gate comparison
├── eval_metrics.py          # Dice + HD95 on the validation split
├── inference.py             # save predictions for visualization
├── visualize.py             # plot CT / GT / prediction
├── quickstart.py            # end-to-end sanity check
├── smoke_test.py            # zero-data smoke test
├── verify_data.py           # dataset verification
├── data_stats.py            # dataset statistics
└── requirements.txt
```

## Method

SF-Gate inserts a frequency-domain branch into each skip connection of
Swin-UNETR. It combines a spatial channel-attention branch with an FFT-based
frequency branch and fuses them with a learned gate.

## Results (20-volume benchmark, single 80/20 split)

| Model | Best DSC (%) | HD95 (voxels) |
|-------|--------------|---------------|
| Swin-UNETR (baseline) | 53.31 | 59.29 |
| **SF-Swin-UNETR (ours)** | **55.34** | 69.08 |

## References

Built on MONAI and the Swin-UNETR architecture:

- Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation
  of Brain Tumors in MRI Images", BrainLes 2021.
- Tang et al., "Self-Supervised Pre-Training of Swin Transformers for 3D Medical
  Image Analysis", CVPR 2022.
- Cardoso et al., "MONAI: An open-source framework for deep learning in
  healthcare", arXiv 2022.
- Ma et al., "Toward data-efficient learning: A benchmark for COVID-19 CT lung
  and infection segmentation", Medical Physics 2021.

## AI usage (Software 3.0) disclosure

This project builds on the authors' previous course project, which already
contained the SF-Swin-UNETR model, the data pipeline, the configuration, and the
training/visualization scaffolding. `proj_code/` updates that base, and the
substantial updates were made with the help of generative AI (Claude), in
particular:
- rewriting the training code (`train.py` and the associated training logic);
- setting up the qualitative-result visualization pipeline (`inference.py`,
  `visualize.py`);
- writing the evaluation and statistics helper scripts (`eval_metrics.py`,
  `data_stats.py`).

The model architecture, data pipeline, and experimental results are the authors'
own. All AI-assisted code was reviewed and validated by the authors, who take
full responsibility for the final implementation.

## License

MIT License
