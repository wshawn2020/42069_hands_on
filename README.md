# AI Threat Detection Demo

This repository contains a demo project for 42069 research methods, showcasing hands-on practices for orchestrating an automated AI-driven threat detection system in a containerized environment.

## Description

This project demonstrates how to set up and run an AI-driven threat detection pipeline using a containerized environment. The pipeline leverages a Random Forest model to detect threats based on the CIC-IDS2017 dataset. The automation is driven by a `launch.sh` script, with container configurations defined in `Dockerfile` and `docker-compose.yml`. Results, such as heatmaps, are exported to the `./results` directory.

## Dependencies

- Bash
- Docker
- `docker-compose`

## Setup

1. Ensure all dependencies (Bash, Docker, `docker-compose`) are installed on your system.

2. Place the `dataset.csv` file in the `./data` directory.

3. Verify that the `random_forest_detector.py` script is in the `./scripts` directory.

4. Run the automation script:

   ```bash
   ./launch.sh
   ```

5. Check the `./results` directory for generated heatmap outputs.

## File Structure

- `./data/dataset.csv`: Input dataset for the threat detection pipeline (CIC-IDS2017).
- `./scripts/random_forest_detector.py`: Python script implementing the Random Forest model for threat detection.
- `./results/`: Directory where heatmap outputs are saved.
- `Dockerfile`: Defines the container environment setup.
- `docker-compose.yml`: Configures the containerized services.
- `launch.sh`: Automation script to orchestrate the pipeline.
- `requirements.txt`: Lists Python dependencies for the project.
- `demo.webm`: Video demonstrating the project workflow.

## Demo Video

A demo video (`demo.webm`) is included in the repository, providing a visual overview of the project's workflow.

## Dataset

The dataset used in this project is based on CIC-IDS2017, a comprehensive dataset for intrusion detection research, provided by the Canadian Institute for Cybersecurity.
