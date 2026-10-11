# Industrial Quality Inspection Using YOLOv8 on Raspberry Pi

## Overview

This project was developed during my internship at Crown Maghreb, part of Crown Holdings, as part of an industrial quality inspection application.

The objective was to use computer vision to automate visual inspection by combining a YOLOv8 object detection model, a Raspberry Pi 4, and a camera. A web interface developed with Flask allows users to interact with the system and view inspection results.

The project explores how deep learning can be deployed on embedded hardware to support quality control in an industrial environment.

## Technologies Used

- **Programming language:** Python
- **Object detection:** YOLOv8
- **Computer vision:** OpenCV
- **Embedded system:** Raspberry Pi 4
- **Camera:** Raspberry Pi HQ Camera
- **Web framework:** Flask

## System Workflow

1. The camera captures images of the products to be inspected.
2. The images are processed and passed to the YOLOv8 model.
3. The model analyzes the images and produces detection results.
4. The results are made available through the Flask web interface.

## Main Features

- Image acquisition using a camera connected to the Raspberry Pi.
- Object detection using a YOLOv8 model.
- Local inference on embedded hardware.
- Web interface for interacting with the inspection system.
- Application of computer vision to industrial quality control.

## Project Demo

A demonstration video will be added to show the system in operation, including image acquisition, model predictions, and the web interface.

**Demo video:** [Watch the demonstration] " https://youtu.be/qgYFSMIrSs8 "

## Installation

Clone the repository:

```bash
git clone https://github.com/tass123-eng/stageCrown.git
cd stageCrown
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies, if a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Make sure the camera is configured correctly and the trained YOLOv8 model is available at the path expected by the application.

Run the application using the appropriate Python entry point for the project.

## Future Improvements

- Improve inference speed on Raspberry Pi.
- Evaluate detection accuracy and processing time.
- Add inspection history and result logging.
- Optimize the model for embedded deployment.

## Author

**Tassnim Ouerghemmi**

Engineering student specializing in Connected Systems, interested in computer vision, deep learning, and embedded artificial intelligence.

[GitHub Profile](https://github.com/tass123-eng)

## Acknowledgements

This project was carried out during my internship at Crown Maghreb, providing an opportunity to apply computer vision and embedded AI techniques to an industrial use case.
