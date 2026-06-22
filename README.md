# Multi-Player Tracking in Football Videos

A comparative honours research project evaluating **DeepSORT** and **ByteTrack** for multi-player tracking in football broadcast videos using **YOLOv8** detections.

This project investigates how modern multi-object tracking algorithms perform in football footage, where challenges include camera movement, player occlusion, similar jerseys, dense formations, and rapid changes in direction.

## Project Overview

Football analytics increasingly depends on accurate player tracking for tactical and performance analysis. Manual annotation is slow, expensive, and error-prone, so this project compares two automated tracking approaches:

* **DeepSORT** — tracking-by-detection with motion modelling and deep appearance embeddings.
* **ByteTrack** — tracking-by-detection using two-stage association with high- and low-confidence detections.

Both trackers are evaluated on football broadcast sequences from the **SoccerNet** dataset using a shared **YOLOv8** detector.

## Research Question

How do DeepSORT and ByteTrack compare when tracking multiple football players in broadcast match footage?

The comparison focuses on:

* Tracking accuracy
* Identity preservation
* False positives and false negatives
* Player recall
* Tracking stability
* Suitability for football analytics workflows

## Key Findings

ByteTrack produced cleaner and more stable tracking results, with higher overall tracking accuracy and near-perfect precision. It is better suited for tactical and statistical football analytics where low-noise player trajectories are important.

DeepSORT preserved identities better and achieved stronger recall, but it produced more false positives and noisier tracks. It is useful when detecting as many player-like entities as possible is more important than avoiding false alarms.

## Results Summary

| Sequence |   Tracker |  MOTA |  IDF1 | Precision | Recall | ID Switches |
| -------- | --------: | ----: | ----: | --------: | -----: | ----------: |
| Seq 1    | ByteTrack | 0.863 | 0.697 |     0.990 |  0.877 |          64 |
| Seq 1    |  DeepSORT | 0.467 | 0.766 |     0.654 |  0.997 |          21 |
| Seq 2    | ByteTrack | 0.884 | 0.700 |     0.997 |  0.891 |          63 |
| Seq 2    |  DeepSORT | 0.722 | 0.800 |     0.785 |  0.997 |          34 |
| Seq 3    | ByteTrack | 0.746 | 0.511 |     0.999 |  0.755 |         121 |
| Seq 3    |  DeepSORT | 0.438 | 0.606 |     0.644 |  0.989 |          58 |

## Interpretation

### ByteTrack

ByteTrack achieved the strongest overall performance for reliable football analytics.

Strengths:

* Higher MOTA scores across all tested sequences
* Very high precision
* Smooth and stable trajectories
* Fewer false positives
* Better suited for tactical analysis pipelines

Limitations:

* More identity switches than DeepSORT
* Can fragment identities when players overlap or cross paths
* Lower recall in some sequences

### DeepSORT

DeepSORT performed better in identity preservation and recall.

Strengths:

* Higher IDF1 scores
* Stronger recall
* Fewer identity switches
* Better at maintaining identities when detections are reliable

Limitations:

* More false positives
* Noisier tracking output
* Can incorrectly track referees, the ball, or non-player objects
* Less reliable for automated downstream analytics

## Tech Stack

* Python 3.11
* OpenCV
* YOLOv8
* DeepSORT
* ByteTrack
* SoccerNet dataset
* motmetrics
* NumPy
* Pandas
* scikit-learn
* Matplotlib
* YAML-based configuration
* Jupyter Notebook demo

## Methodology

The project follows a modular tracking-by-detection pipeline:

```text
Football Video
     ↓
YOLOv8 Player Detection
     ↓
DeepSORT / ByteTrack Tracking
     ↓
MOT Metrics Evaluation
     ↓
Comparison and Visualisation
```

The system is split into three main components:

1. **Detection Module**
   YOLOv8 is used to detect players in each frame.

2. **Tracking Module**
   DeepSORT and ByteTrack are run separately on the same detections to ensure fair comparison.

3. **Evaluation Module**
   Tracking outputs are evaluated using standard multi-object tracking metrics.

## Evaluation Metrics

The following MOT metrics were used:

| Metric      | Description                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------- |
| MOTA        | Overall tracking accuracy, accounting for false positives, false negatives, and identity switches |
| MOTP        | Bounding box localisation precision                                                               |
| IDF1        | Identity preservation score                                                                       |
| Precision   | Proportion of tracker outputs that correctly match real players                                   |
| Recall      | Proportion of real players successfully detected and tracked                                      |
| ID Switches | Number of times a tracked identity changes incorrectly                                            |
| FPS         | Processing speed for real-time applicability                                                      |

## Dataset

The experiments use football broadcast footage from the **SoccerNet** dataset.

The selected sequences include common football tracking challenges:

* Dense player groupings during set pieces
* Fast counter-attacks
* Broadcast camera movement
* Occlusions
* Similar player appearances
* Varying stadium lighting

> Note: SoccerNet data is not included in this repository. Users must obtain access through the official SoccerNet channels.

## Installation

Clone the repository:

```bash
git clone https://github.com/JustTryingToDoBetter/honours-project.git
cd honours-project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the core dependencies:

```bash
pip install ultralytics opencv-python deep-sort-realtime motmetrics numpy pandas scipy scikit-learn matplotlib pyyaml
```

## Usage

Run DeepSORT tracking:

```bash
python run_deepsort.py --config configs/deepsort.yaml
```

Run ByteTrack tracking:

```bash
python run_bytetrack.py --config configs/bytetrack.yaml
```

Run the demo notebook:

```bash
jupyter notebook
```

Then open the demo notebook and run the full pipeline to reproduce the comparison videos and metrics.

## Suggested Project Structure

```text
honours-project/
├── configs/
│   ├── deepsort.yaml
│   └── bytetrack.yaml
├── data/
│   ├── raw/
│   ├── detections/
│   └── ground_truth/
├── notebooks/
│   └── demo.ipynb
├── outputs/
│   ├── videos/
│   ├── metrics/
│   └── plots/
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── evaluation/
│   └── visualisation/
├── run_deepsort.py
├── run_bytetrack.py
└── README.md
```

## Reproducibility Notes

To reproduce the results:

1. Download the required SoccerNet sequences.
2. Place videos and annotations in the expected data directories.
3. Run YOLOv8 detections or load detections in MOTChallenge format.
4. Run each tracker separately using the provided scripts.
5. Evaluate outputs using the MOT metrics pipeline.
6. Compare metrics and visual tracking outputs.

## Limitations

This project compares general-purpose MOT algorithms in a football-specific setting. While the results show clear trade-offs, both trackers have limitations:

* Similar jerseys make long-term identity preservation difficult.
* Broadcast camera motion can disrupt temporal consistency.
* Occlusions during set pieces reduce detection reliability.
* Non-player objects can be incorrectly tracked.
* Generic person re-identification models are not fully adapted to football footage.

## Future Work

Future improvements could include:

* Jersey number recognition for stronger re-identification
* Team-colour clustering to separate players by team
* Football-specific YOLOv8 fine-tuning
* Hyperparameter tuning for match footage
* More SoccerNet sequences across different match conditions
* Real-time deployment benchmarking
* Ball tracking integration
* Tactical analytics outputs such as heatmaps and player movement summaries

## Research Context

This project was completed as an honours research project at the **University of the Western Cape**.

**Title:** Multi-Player Tracking in Football Videos: A Comparative Study of DeepSORT and ByteTrack
**Author:** J. T. Morrison
**Institution:** University of the Western Cape
**Year:** 2025

## Project Website

Research site:

```text
https://sites.google.com/myuwc.ac.za/tracking-in-football/home
```

## Conclusion

ByteTrack provides the stronger foundation for reliable football player tracking pipelines because of its high precision, cleaner trajectories, and better overall tracking accuracy. DeepSORT remains useful where identity preservation and maximum recall are prioritised, but its tendency to produce false positives makes it less suitable for fully automated tactical analytics.

Together, the comparison shows that football player tracking requires domain-specific adaptation beyond general-purpose MOT algorithms.
