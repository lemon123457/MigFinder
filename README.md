# MigFinder

MigFinder is a platform for automated migrasome analysis in fluorescence microscopy images and time-lapse recordings. It brings together denoising, detection and tracking, with interactive annotation, quantitative measurements and an LLM Agent for analysis.

## Functions

- **Denoise:** reduce image noise before analysis.
- **Detect:** locate migrasomes in raw or denoised images.
- **Annotate:** add, delete, move and resize detection boxes.
- **Track:** follow migrasomes across a continuous image sequence.
- **Measure:** quantify migrasome number, fluorescence intensity and equivalent diameter.
- **Fine-tune:** adapt the detector to independently annotated training, validation and test datasets.
- **LLM Agent:** use natural-language requests to plan analyses, run tools and generate plots.
- **Export:** save images, annotations, measurements and plots.

## Get started

### 1. Local installation

#### Dependencies

MigFinder relies on the following excellent packages:

- [PyTorch and torchvision](https://pytorch.org/)
- [rfdetr](https://github.com/roboflow/rf-detr), [Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft)
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [pandas](https://pandas.pydata.org/)
- [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.github.io/) and [tifffile](https://github.com/cgohlke/tifffile)
- [Matplotlib](https://matplotlib.org/) and [openpyxl](https://openpyxl.readthedocs.io/)

Download the repository and open a terminal in its root folder. Create a Python 3.12 environment and install the CUDA-enabled dependencies:

```console
conda create -n migfinder python=3.12 -y
conda activate migfinder
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-local.txt
```

This configuration uses CUDA 12.8 PyTorch wheels and requires a compatible NVIDIA GPU and driver.

#### Local training

Prepare a single-class COCO detection dataset:

```text
dataset/
  train/    # Training images and _annotations.coco.json
  valid/    # Validation images and _annotations.coco.json
  test/     # Test images and _annotations.coco.json
```

Run the training script with your dataset and output folders:

```console
python training/train.py --dataset "/path/of/dataset" --output "/path/of/results/training"
```

Replace both example paths with your own. The script also accepts `--epochs`, `--batch-size` and `--resolution`; its defaults are 100, 8 and 1008.

#### Local inference

Use the supplied weights in `checkpoint/denoise.pth` and `checkpoint/detect.pth`. These must be the actual model files. If you cloned a repository that stores the weights with Git LFS, run `git lfs pull` first.

Place the frames from **one continuous recording** in a folder, with names such as `frame_0001.tif`, `frame_0002.tif`, and so on. The demo reads images directly from that folder and sorts them in natural filename order. It also accepts a single 2D image. Export multi-page TIFFs as individual frames before using this demo.

After installing the dependencies above, run this command from the downloaded repository folder:

```console
python script/denoise_detect_track_demo.py --input "/path/of/images" --output "/path/of/results"
```

Replace `/path/of/images` with your image or frame-folder path and `/path/of/results` with a **new or empty** output folder. On Windows, quoted paths such as `"D:/data/frames"` and `"D:/results/migfinder_demo"` work the same way. The demo automatically locates the repository root, its denoise, detect and tracker modules, and the default checkpoints; no `PYTHONPATH` setting or source-code editing is needed.

For a short trial, add `--max-frames 3`. Use `--no-track` for unrelated images. To adjust detection confidence or model input size, add `--threshold 0.25` or `--resolution 1120`; resolution must be a positive multiple of 56. The default device is CUDA.

Results include denoised images, annotated overlays, per-object `detections.csv`, confirmed-track observations in `tracks.csv`, a `frame_summary.csv`, and `run_config.json`. Single-image and `--no-track` runs still save detections and measurements; their track table contains only its header. Intensity is measured from the original image, while mask area and equivalent diameter are measured inside the denoised detection region in pixels. Track IDs can remain unconfirmed for newly appearing objects, so detection count and active-track count can differ.

Use `--denoise-weights` and `--detect-weights` to select other checkpoint files. Run `python script/denoise_detect_track_demo.py --help` for all options.

#### Figure analysis scripts

The scripts under `analysis/` reproduce selected paper analyses and need the original image, annotation or feature-table inputs. These inputs are not included in this code repository. Install their additional dependencies with `python -m pip install -r requirements-analysis.txt` after completing the local installation above. Replace `/path/of/...` examples with your own paths or use each script's command-line options where available.

The included Figure 5 script reads the `track_features` sheet from an external `figure5_source_data.xlsx` file and generates S1/S2 assignments for panels b–d. The Figure 6 script generates detection outputs for panels b, g and l. These scripts do not regenerate every panel of their respective figures.

### 2. GUI

#### System requirements

The desktop application runs on **Windows with an NVIDIA CUDA GPU**. Denoising, detection, model fine-tuning and local LLM generation require the GPU.

Reference workstation:

| Component | Configuration |
|---|---|
| Operating system | Windows 11 Pro, 64-bit |
| CPU | Intel Core i5-12400F |
| RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 4060 |
| GPU memory | 8 GB |

This is a reference configuration, rather than a minimum requirement.

Clone the complete repository, open `GUI`, and double-click `GUI/MigFinder.exe`. Keep the complete `GUI/data` folder beside the launcher. First launch automatically prepares the included offline runtime and Qwen model; later launches reuse the local cache. No Python or separate model installation is required for the GUI. See [GUI/USER_GUIDE.md](GUI/USER_GUIDE.md) for installation and launch instructions.

#### Basic workflow

1. **Import:** choose **Files…** or **Folder…** to load images or one continuous time-lapse sequence.
2. **Denoise:** reduce noise if required.
3. **Detect:** select **Raw** or **Denoised**, set the confidence threshold (default `0.25`), then detect migrasomes.
4. **Annotate:** use **Draw**, **Delete**, **Select** and **Undo** to add or correct boxes.
5. **Track:** associate existing boxes across frames. Run tracking again after changing annotations.
6. **Save results:** export the current images, COCO annotations, measurement tables and plots.

Each operation runs when selected: Detect uses the chosen image source, Track uses existing boxes, and Save results exports the current state. Single-image analysis does not require tracking.

#### Fine-tuning and LLM Agent

Open **Fine-tune**, select separate **Train**, **Val** and **Test** COCO folders, choose a model-output folder and start training. The application evaluates the new detector against the original detector on the supplied test set. Use **Use model** to select the new weights for subsequent detection.

Use the **LLM Agent** to request quantitative analyses, plots or exports. For example:

> Plot the mean raw intensity and equivalent diameter over the video frames.

Save and reuse analyses with **Save Plan / Load Plan / Run Plan**. If the agent prepares a Python draft, inspect it in **Review Python**, then choose **Approve and run** to execute it.
