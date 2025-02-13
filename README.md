# Experiment Setup and Execution Guide

# Introduction

This repository contains a PsychoPy-based experiment that involves both visual and auditory stimuli. Participants are required to observe flickering gratings and listen to amplitude-modulated tones while responding to stimulus conditions.

# Requirements

The experiment requires the following:

**Python Version**: 3.9

**Libraries**:

- PsychoPy (2024.2.4)
- NumPy
- SoundDevice
- wxPython
- dukpy
- ParallelPort (Windows/Linux only)

For macOS users, `ParallelPort` functionality is not available and will be bypassed.

# Installation

This project relies on a Conda environment for managing dependencies. Before proceeding, ensure that Conda is installed on your system (via Anaconda, Miniconda, or Mamba). Once Conda is set up, you can create the required environment using the following command:

```
conda env create -f environment.yml
```

This will set up a `neuronal-computations` environment with all necessary dependencies.

# Running the Experiment

## 1. Activate the Conda Environment

```
conda activate neuronal-computations
```

## 2. Running the Experiment

Navigate to the experiment directory (example):

```
cd C:\Users\BryanTan\Documents\experiment
```

Then, execute the experiment script:

```
python stimuli_final.py
```

## 3. Common Parameters

You can modify common parameters directly in `ExperimentConfig` within `stimuli_final.py`:

`NUM_BLOCKS`: Number of experiment blocks (default: 6)

`TRIALS_PER_BLOCK`: Trials per block (default: 40)

`MOD_FREQ_HIGH`: High Modulation frequency (default: 450 Hz)

`MOD_FREQ_LOW`: Low Modulation frequency (default: 400 Hz)

`VISUAL_FREQ_HIGH`: High flicker frequency (default: 15 Hz)

`VISUAL_FREQ_LOW`: Low flicker frequency (default: 12 Hz)

`RESTING_DURATION`: Duration of each state in the resting state phase (default: 60 seconds)

`TRAINING_DURATION`: Duration of each training in the training session (default: 15 seconds)

`BREAK_TIME`: Duration of breaks between blocks (default: 60 seconds)

`RUN_RESTING_STATE` : Switch for the resting state phase (default: True)

`RUN_TRAINING_SESSION` : Switch for the training session (default: True)

To modify these, open `stimuli_final.py` and adjust the values before running the experiment.

# Code Explanation

This section provides a breakdown of the key components of the experiment code:

## 1. Experiment Configuration (`ExperimentConfig`)

Defines global parameters such as sample rate, frequencies, durations, and refresh rate.

## 2. `AudioManager`

Manages the creation of amplitude-modulated tones and normalizes audio output.

## 3. `VisualStimulus` and `AuditoryStimulus`

- `VisualStimulus`: Generates flickering gratings at specified frequencies.
- `AuditoryStimulus`: Generates stereo audio tones and plays them through `sounddevice`.

## 4. `BittiumTriggerSender`

Handles sending external triggers to hardware using the parallel port (Windows/Linux only). For macOS, it prints trigger events for debugging.

## 5. Experiment Flow

- **Pre-Experiment Checklist**: Ensures the participant is ready.
- **Resting State**: Participants relax with eyes open/closed.
- **Training Sessions**: Familiarizes participants with stimuli.
- **Experiment Trials**: Participants respond to combined audio-visual stimuli.
- **Breaks & Feedback**: After blocks of trials, participants take breaks and see performance summaries.

## 6. Multithreading for Synchronization

The code uses `threading` to run auditory and visual stimuli simultaneously, ensuring precise timing.
