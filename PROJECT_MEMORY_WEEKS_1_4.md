# Ultrasound Project Memory: Weeks 1–4

Last consolidated: 4 November 2026

## Project identity

- Primary model: `google/medgemma-1.5-4b-it`
- Exact model revision: `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`
- Main Google Drive root: `/content/drive/MyDrive/Ultrasound_Project-Siyona/`
- Main environment: Google Colab with a Tesla T4 GPU
- Model loading: 4-bit quantization using `bitsandbytes`
- Hugging Face account used in the notebooks: `siyonanaik`

Recorded package versions from the completed experiments:

- pandas `2.2.3`
- Pillow `11.3.0`
- PyTorch `2.11.0+cu130`
- Transformers `5.20.2` during the recorded completed work; later Colab grounding recovery used `5.22.0`
- Accelerate `1.15.0`
- bitsandbytes `0.50.2`
- scikit-learn `1.9.1`
- seaborn `0.13.2`

## Authoritative notebooks

- Week 1 setup: `ColabNotebooks/00_setup.ipynb`
- Week 1 data preparation: `ColabNotebooks/01_data.ipynb`
- Week 2 BUSI experiments: `ColabNotebooks/02_busi_inference.ipynb`
- Week 2 CAMUS experiments: `ColabNotebooks/02_camus_inference.ipynb`
- Week 3 completed BUSI classification: `ColabNotebooks/03_busi_classification_completed.ipynb`
- Week 4 completed BUSI grounding: `ColabNotebooks/04_busi_grounding.ipynb`

Use `03_busi_classification_completed.ipynb` as the final Week 3 notebook. An older version had a Colab remote-save conflict.

## Week 1: Environment, BUSI preparation, fixed split, and first inference

### Environment and first model test

- Confirmed a Tesla T4 GPU in Colab.
- Installed Transformers, Accelerate, bitsandbytes, Hugging Face Hub, and Pillow.
- Logged in to Hugging Face and successfully loaded MedGemma in 4-bit mode.
- Ran a first inference on `benign (1).png` using the BUSI classification prompt.
- The model returned valid JSON, `{"label": "normal"}`, but the true class was benign, so the first prediction was incorrect.
- Saved the first result to `results/raw/first_medgemma_result.csv` in Drive.

### BUSI dataset preparation

- Original archive: `data/raw/BUSI/BUSI_dataset.zip`
- Extracted under: `data/raw/BUSI/extracted/`
- Clean ultrasound images: 780
- Mask files: 798
- Clean-image class counts:
  - benign: 437
  - malignant: 210
  - normal: 133
- Mask files were excluded from the clean-image list.

Created `data/busi_manifest.csv` with:

- `image_id`
- `true_label`
- `image_path`
- `number_of_masks`
- `mask_paths`

Multiple mask paths are separated with semicolons.

### Fixed BUSI split

The split used:

```python
random_state=42
stratify=manifest_df["true_label"]
```

Saved files:

- `data/splits/busi_development_20.csv`
- `data/splits/busi_test_760.csv`

Development distribution:

- benign: 11
- malignant: 5
- normal: 4

Test distribution:

- benign: 426
- malignant: 205
- normal: 129

The splits contain no overlapping image IDs and together cover all 780 manifest rows.

### Fixed BUSI classification prompt

Saved at `prompts/busi_classification_v1.txt`:

```text
Classify this breast ultrasound image as exactly one of:
normal, benign lesion, malignant lesion, or insufficient image quality.

Return JSON only using this format:
{"label": "<allowed label>"}
```

## Week 2: Development experiments on BUSI and CAMUS

### BUSI 20-image dry run

Notebook: `02_busi_inference.ipynb`

Fixed greedy settings:

- `do_sample=False`
- `temperature=None`
- `max_new_tokens=60`
- random seed `42`

Results:

- Correct predictions: 5/20
- Accuracy: 25%
- Valid JSON rate: 100%
- Average latency: about 3.02 seconds per image

Saved result:

- `results/dry_run/busi_development_20_medgemma.csv`

### BUSI temperature experiment

The same 20 images and prompt were tested with greedy decoding and sampling temperatures `0.1`, `0.3`, `0.7`, and `1.0`.

| Condition | Accuracy | Valid JSON | Average latency |
|---|---:|---:|---:|
| Greedy | 25% | 100% | 3.02 s |
| Temperature 0.1 | 25% | 100% | 3.29 s |
| Temperature 0.3 | 15% | 65% | 4.61 s |
| Temperature 0.7 | 10% | 55% | 4.94 s |
| Temperature 1.0 | 15% | 40% | 4.20 s |

Decision: greedy decoding was retained because it tied for the best accuracy, was faster, fully format-compliant, and deterministic. Temperature has no meaningful effect when `do_sample=False`.

Saved under `results/temperature_experiment/`, including:

- `busi_temperature_experiment_medgemma.csv`
- `busi_temperature_summary.csv`
- `busi_greedy_vs_temperature_comparison.csv`
- `busi_confidence_proxies.csv`
- `busi_invalid_responses.csv`

Additional BUSI prompt variants were saved but not run through MedGemma:

- `prompts/busi_prompt_variants/busi_morphology_v1.txt`
- `prompts/busi_prompt_variants/busi_image_quality_v1.txt`
- `prompts/busi_prompt_variants/busi_lesion_presence_v1.txt`
- `prompts/busi_prompt_variants/busi_classification_abstention_v1.txt`

### Confidence proxies

Calculated indirect measures because the text pipeline did not return calibrated label probabilities:

- `format_reliability_rate`
- `valid_label_agreement`
- `overall_modal_support`
- `correct_run_rate`

Interpretation rule: stability measures decoding consistency, not true diagnostic confidence. A model can be consistently wrong.

### CAMUS development experiment

Notebook: `02_camus_inference.ipynb`

Dataset location:

- `data/raw/CAMUS/CAMUS_public/`
- Contains `database_nifti/` and `database_split/`
- Official validation split: 50 patients

Fixed development manifest:

- 20 PNG images from 10 unique validation patients
- 10 A2C and 10 A4C
- 10 end-diastolic and 10 end-systolic
- 8 good-quality, 8 medium-quality, and 4 poor-quality
- Saved as `data/splits/camus_development_20.csv`
- Converted PNGs saved under `data/processed/camus_development_20/`
- Images were transposed so the narrow sector point is at the top and the wider sector extends downward.

Fixed multi-task prompt:

- `prompts/camus/camus_multitask_classification_v1.txt`
- Requested `view`, `image_quality`, and `cardiac_phase` in JSON.

CAMUS decoding results:

| Condition | Complete validity | View accuracy | Quality accuracy | Phase accuracy | Complete accuracy |
|---|---:|---:|---:|---:|---:|
| Greedy | 100% | 50% | 40% | 50% | 10% |
| Temperature 0.1 | 95% | 50% | 40% | 50% | 10% |
| Temperature 0.3 | 70% | 45% | 30% | 30% | 10% |
| Temperature 0.7 | 65% | 30% | 30% | 30% | 5% |
| Temperature 1.0 | 20% | 5% | 0% | 0% | 0% |

Important finding: MedGemma often repeated `A2C | good | end diastole`. High agreement therefore did not imply correctness.

Saved under `results/camus_temperature_experiment/`, including:

- `camus_complete_experiment_medgemma.csv`
- `camus_decoding_summary.csv`
- `camus_confidence_proxies.csv`
- `camus_invalid_responses.csv`

Task-specific CAMUS prompt variants were saved but not run:

- `camus_view_classification_v1.txt`
- `camus_image_quality_v1.txt`
- `camus_cardiac_phase_v1.txt`

## Week 3: Full BUSI clean-image classification

Notebook: `03_busi_classification_completed.ipynb`

Evaluation configuration:

- 760 untouched BUSI test images
- Fixed BUSI classification prompt
- Greedy decoding
- `do_sample=False`
- `temperature=None`
- `max_new_tokens=60`
- random seed `42`
- Checkpoint every 20 images

Completion checks:

- Completed: 760
- Remaining: 0
- Duplicate image IDs: 0

Overall results:

- Correct: 237/760
- Accuracy: 31.18%
- Valid-label responses: 742/760
- Valid-label rate: 97.63%
- Invalid responses: 18
- Insufficient-image-quality predictions: 209
- Average latency: about 3.21 seconds
- Macro F1: 28.70%
- Weighted F1: 32.76%

Per-class metrics:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Normal | 27.69% | 69.77% | 39.65% |
| Benign lesion | 71.01% | 34.51% | 46.45% |
| Malignant lesion | 0% | 0% | 0% |

Prediction distribution:

- normal: 325
- benign lesion: 207
- malignant lesion: 1
- insufficient image quality: 209
- invalid response: 18

The only malignant prediction was for a benign image. None of the 205 malignant images were correctly identified.

Confusion matrix counts:

| True / Predicted | Normal | Benign | Malignant | Insufficient quality | Invalid |
|---|---:|---:|---:|---:|---:|
| Normal | 90 | 1 | 0 | 38 | 0 |
| Benign | 180 | 147 | 1 | 92 | 6 |
| Malignant | 55 | 59 | 0 | 79 | 12 |

Majority-class baseline:

- Majority class: benign lesion
- Baseline accuracy: 56.05%
- MedGemma accuracy: 31.18%
- Difference: -24.87 percentage points

Conclusion: MedGemma performed below the majority baseline and had 0% malignant recall.

Saved under `results/busi_full_classification/`, including:

- `busi_test_760_greedy_medgemma.csv`
- `busi_full_evaluation_summary.csv`
- `busi_per_class_metrics.csv`
- `busi_prediction_distribution.csv`
- `busi_majority_baseline_comparison.csv`
- `busi_class_error_analysis.csv`
- `busi_full_invalid_responses.csv`
- `busi_confusion_matrix_greedy.csv`
- `busi_confusion_matrix_greedy.png`
- `busi_development_decoding_comparison.csv`

## Week 4: Full BUSI grounding evaluation

Notebook: `04_busi_grounding.ipynb`

### Ground-truth box method

- Confirmed `mask_paths` uses semicolons.
- Mask-count distribution across BUSI:
  - one mask: 763 images
  - two masks: 16 images
  - three masks: 1 image
- Multiple masks are combined using a logical OR union.
- One enclosing ground-truth box is calculated around all white pixels in the union.
- Pixel-box format: `[x_min, y_min, x_max, y_max]`.
- Minimum coordinates are inclusive; maximum coordinates are exclusive.
- Normal images have empty masks and therefore no target box.
- IoU is calculated only for benign and malignant images. A missing or unusable predicted box on a lesion image receives IoU 0.

### Grounding prompt and output

Saved at `prompts/busi_grounding_v1.txt`.

The model was asked to return:

```json
{"lesion_present": true, "box": [x_min, y_min, x_max, y_max]}
```

or:

```json
{"lesion_present": false, "box": null}
```

Coordinates use a normalized 0–1000 scale. The parser removes optional Markdown code fences before validating the JSON.

### Development grounding results

- Development images: 20
- Lesion images eligible for IoU: 16
- Normal images with no target box: 4
- Valid response rate: 100%
- Lesion-presence accuracy: 80%
- Mean lesion-image IoU: 0.295814
- Median lesion-image IoU: 0.271914
- IoU at least 0.25: 50%
- IoU at least 0.50: 25%
- Average latency: 5.956568 seconds

Development finding: MedGemma predicted a lesion in every image, including all four normal images, and frequently reused broad normalized boxes such as `[100, 100, 800, 800]`.

### Full test grounding results

Completion checks:

- Test predictions: 760
- Ground-truth rows: 760
- Duplicate image IDs: 0
- Missing image IDs: 0
- Visual examples: 10
- All required files verified as present

Overall results:

- Valid response rate: 100%
- Lesion-presence accuracy: 83.0263%
- Mean IoU on 631 lesion images: 0.239125
- Median IoU: 0.205101
- IoU at least 0.25: 42.4723%
- IoU at least 0.50: 13.1537%
- Average latency: 5.907653 seconds

Per-class results:

| True class | Images | Predicted-lesion rate | Presence accuracy | Mean IoU | Median IoU | IoU >= 0.25 | IoU >= 0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Benign | 426 | 100% | 100% | 0.175967 | 0.117498 | 30.0469% | 4.6948% |
| Malignant | 205 | 100% | 100% | 0.370370 | 0.361067 | 68.2927% | 30.7317% |
| Normal | 129 | 100% | 0% | N/A | N/A | N/A | N/A |

Main grounding finding: MedGemma predicted a lesion for every test image. This gives perfect lesion-presence sensitivity on benign and malignant images but 0% specificity on normal images. Grounding overlap was limited overall and was better for malignant than benign images.

Saved under `results/busi_grounding/`, including:

- `busi_development_grounding_medgemma.csv`
- `busi_development_grounding_summary.csv`
- `busi_test_ground_truth_boxes.csv`
- `busi_test_grounding_medgemma.csv`
- `busi_test_grounding_summary.csv`
- `busi_test_grounding_per_class.csv`
- `busi_grounding_10_visual_examples.png`
- `busi_grounding_10_visual_examples.csv`

The ten visual examples contain four benign images spread across low-to-high IoU, four malignant images spread across low-to-high IoU, and two normal images. Green boxes are ground truth and red boxes are MedGemma predictions.

## Decisions that must remain fixed

- Do not mix the 20 development images into the 760-image final metrics.
- Do not rerun completed Week 2, Week 3, or Week 4 inference unless a new experiment explicitly requires it.
- Use greedy decoding as the selected final decoding strategy.
- Keep the exact model revision recorded with results.
- Preserve raw model responses, including invalid responses.
- Treat confidence proxies as stability or formatting measures, not calibrated diagnostic confidence.
- Treat normal BUSI grounding images as having no target box; do not assign them an IoU.
- For multi-mask BUSI images, use the union of all masks and one enclosing box.

## Overall conclusion through Week 4

The experimental pipeline is reproducible and technically complete through BUSI classification and grounding. MedGemma followed structured formats reliably under greedy decoding, but clinical performance was weak:

- BUSI classification was below the benign-majority baseline.
- Malignant classification recall was 0%.
- Grounding produced valid structured responses but predicted a lesion in every image.
- Grounding overlap was modest, with mean IoU 0.239 across lesion images.
- Higher decoding temperatures consistently reduced format reliability and accuracy.
- High response agreement can reflect a stable failure mode rather than genuine confidence.
