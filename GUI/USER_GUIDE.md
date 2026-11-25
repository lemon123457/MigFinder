# MigFinder User Guide

## Open MigFinder

1. Open the repository's **GUI** folder.
2. Double-click **MigFinder.exe**.
3. On first launch, wait for the preparation window to verify and unpack the
   included files. No models or libraries are downloaded. The runtime is stored
   under `%LOCALAPPDATA%\MigFinder\portable_runtime`.
4. MigFinder opens automatically. Later launches reuse the prepared runtime.

## Main workflow

1. **Import → Files… / Folder…:** Select images, a folder of consecutive frames, a multi-page TIFF, or a video. You can type or paste a path into the file dialog. Folder import reads directly contained images and videos without searching subfolders. Files imported together are naturally sorted by filename. New imports are appended to the list; right-click an item on the left to use **Remove / Clear all**. RGB TIFF previews are generated after import; the original files are preserved.
2. **Denoise:** Reduce image noise.
3. **Detect:** Process each image independently, using the current **Raw / Denoised** selection. **Threshold** sets the confidence cutoff: the default is `0.25`, with a range of `0.01–1.00`. Click **Detect** again after changing the threshold. Rerunning detection replaces existing boxes. The single-frame results table shows **Object**, **X (px)** and **Y (px)**; coordinates refer to the box centre, with the origin at the top-left corner of the image.
4. **Track:** Treat all frames in the imported list as one continuous sequence. The results show confirmed **Track** IDs, **Raw intensity** and **Diameter (pixel)** for the current frame.
5. **Save results:** Select an output location and save the results currently available.

## Box editing and measurements

Choose **Select** and click a box. Drag one of its eight corner/edge handles to resize it, or drag inside the box to move it. Use **Draw / Delete / Undo** to add a box, delete a box or undo an edit. Scroll to zoom and choose **Fit** to fit the image to the window. **Raw / Denoised** changes the displayed image and selects the input for **Detect**. **Masks** displays binary measurement regions. Use **Play** and the slider to preview consecutive frames.

## Saved files

Each save creates a new `MigFinder_<timestamp>` folder:

| Contents | Files |
|---|---|
| RGB uint8 TIFFs, when denoising has been completed | `denoised/*_denoised.tif` |
| Original pixel data, preserving bit depth and RGB/grayscale properties | `images/*.tif` |
| COCO annotations and matching training-input images | `coco/_annotations.coco.json`, `coco/images/*.png` |
| Detection, intensity, diameter and coordinate measurements | `object_measurements.csv` |
| Per-image counts, sequence totals and track summaries | `image_summary.csv`, `sequence_summary.csv`, `track_summary.csv` |
| Images with current boxes and track IDs | `overlays/*.png` |
| Count, original-image intensity and diameter plots | `plots/*.png` |

## Independent model fine-tuning

1. Prepare three independent image sets: **train**, **val** and **test**. Annotate all boxes for each set in the main window, then click **Save results**. Each export provides a `coco` image/JSON folder.
2. Open **Fine-tune**. Click **Browse…** in the **Train / Val / Test** rows and select the corresponding `coco` folders. Common single-class COCO folders are also supported: annotations may be named `_annotations.coco.json` or `annotations.json`, with images beside the JSON or in an `images` subfolder. Labels should describe migrasome bounding boxes.
3. Set **Learning rate**, **Epochs** and **Resolution**. Choose an output location under **Save model to**, then click **Start fine-tuning**.
4. A separate timestamped subfolder is created in the selected output location. It contains a training-data snapshot, model weights, epoch-level metrics, independent test metrics for the original and new detectors, and comparison plots. Click **Open results** to inspect them. **Use model** selects the new weights for subsequent detection; this selection is retained after restarting the application.

## LLM Agent and conversation management

Use the conversation selector on the right to switch histories. **New chat** creates a conversation and **Delete chat** deletes the current one. Each conversation stores its own history, latest executable plan and response language. Switching conversations does not change the images imported in the main window. The conversation list is preserved after restart. Deleting the final conversation creates a new empty one automatically.

The application includes a local Qwen3-4B-Instruct GGUF model. Click **Send** or press **Enter** to send a message; press **Shift+Enter** to insert a new line.

Valid analysis plans run automatically. For example:

> Count the existing tracks and save the images, annotations, overlays and plots.

The LLM Agent calls the available detection, tracking, statistics, plotting and export tools, then returns measured counts and file links. General questions use the same conversation context. Clicking **Save results** on the toolbar does not add a chat reply; when you explicitly request a save through the LLM Agent, it reports the execution result in the conversation.

**Save Plan** saves the latest executable plan in the current conversation. **Load Plan** opens a saved plan for inspection; choose **Run Plan** to reuse it on the current images. A new conversation does not inherit another conversation's plan. General questions do not overwrite the current conversation's latest executable plan. Fine-tuning runs in the separate **Fine-tune** window.

## Reviewing and running Python drafts

For example, ask:

> Write Python to calculate the mean raw intensity, save a summary CSV and plot a histogram. Prepare a draft for review.

1. The LLM Agent prepares a Python draft and displays **Review Python**. Preparing a draft does not execute it.
2. Click **Review Python** to read or edit the complete code, then choose **Save results to**. You can also choose **Save script** or **Cancel** to defer execution.
3. Click **Approve and run** to start a separate background process. Approval applies to the current conversation, the reviewed code and its output folder. Changed code, cancelled review, missing approval or reuse of an already consumed approval cannot execute the draft.
4. On success, the application returns the results folder and links to the CSV files, images and other outputs actually created.
