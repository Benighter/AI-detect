# AI Object Detection Web App

## Description
This project is a React-based web application that performs real-time object detection using TensorFlow.js and the COCO-SSD (Common Objects in Context - Single Shot Detector) model. It also includes a custom training feature that allows users to add and detect their own object classes.

## Features
- Real-time object detection using device camera
- Pre-trained detection for 80+ object classes using COCO-SSD
- Custom object class training and detection
- Live mode and Detect-and-Save mode
- FPS (Frames Per Second) counter in live mode
- Object count statistics
- Detailed view of detected objects

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Node.js (v12.0 or later)
- npm (usually comes with Node.js)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ai-detection-app.git
   cd ai-detection-app
   ```

2. Install the dependencies:
   ```
   npm install
   ```

## Usage

1. Start the development server:
   ```
   npm start
   ```

2. Open your web browser and navigate to `http://localhost:3000`

3. Grant camera permissions when prompted

4. The app will start in live detection mode. You can switch between live mode and detect-and-save mode using the button provided.

5. To add a custom class:
   - Enter a name for your new class in the input field
   - Click "Add Class"
   - Upload one or more images for this class using the file input that appears

6. The app will now attempt to detect your custom objects alongside the pre-trained COCO-SSD objects.

## How It Works

- The app uses TensorFlow.js to load and run the COCO-SSD model in the browser.
- For custom object detection, it uses a simple image similarity comparison.
- In live mode, the app continuously captures frames from the camera and runs object detection on each frame.
- In detect-and-save mode, you can capture a single frame for analysis.

## Limitations

- Custom object detection is based on whole-image similarity, not localized object detection.
- Training data (uploaded images) is stored in memory and will be lost when the page is refreshed.
- Performance may vary depending on the device's capabilities.

## Future Improvements

- Implement proper storage for training data (e.g., IndexedDB or server-side storage)
- Add more advanced custom training capabilities
- Improve custom object detection accuracy
- Add ability to export and import trained custom models

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow.js team for providing the tools and models
- React team for the excellent web application framework

## Contact

If you have any questions or feedback, please open an issue in the GitHub repository.

Happy detecting!