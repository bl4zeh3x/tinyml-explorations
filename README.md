# TinyML Explorations

Hands-on work in TinyML — training, quantizing, and deploying neural networks on microcontrollers with severe resource constraints.

The core problem: a typical microcontroller has no GPU, often under 256KB of RAM, runs on a coin cell battery, and must make real-time decisions. Getting useful machine learning to work under those constraints requires completely different thinking than cloud ML. This repository documents that process.

## Active Goal

**Keyword spotting system** — a model that runs continuously on a microcontroller, detects a specific wake word in an audio stream, and fits within 50KB of flash memory.

## Repository Structure

* experiments/ — training notebooks, quantization experiments, benchmark results
* deployments/ — firmware and hardware deployment code
* notes/ — paper summaries, architecture notes, learning log

## Technical Environment

| Layer               | Tool                                 | Reason                                  |
| ------------------- | ------------------------------------ | --------------------------------------- |
| Model training      | Kaggle Notebooks                     | Free GPU access                         |
| ML framework        | TensorFlow Lite for Microcontrollers | MCU deployment                          |
| Deployment pipeline | Edge Impulse                         | Data collection and firmware generation |
| Target hardware     | ESP32 / Arduino Nano 33 BLE Sense    | TF Lite Micro support                   |
| Local development   | WSL2 + Ubuntu + Conda + VS Code      | Reproducible environment                |

## Progress Log

| Date       | Update                                                                  |
| ---------- | ----------------------------------------------------------------------- |
| 2026-06-16 | Repository initialized. Environment operational. Research phase begins. |
