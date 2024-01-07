# Track My Form

This project utilizes TensorFlow's Pose Detection model, specifically [MoveNet](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection), to track user's exercise form and count repetitions.

## Features and Guide

- **Toggle AI and Webcam**
    - Toggling AI will run and visualize the Pose Detection model's predictions from the user's webcam
    - Toggling Webcam toggles the visibility of the Web Cam
    - The AI predictions can be visualized while the webcam is toggled off

- **Tracing Keypoint Path**
    - Clicking on red keypoints that are detected by the model will select them for tracking and turn the point green
    - Clicking a selected green point will unselect it and turn it back to red
    - After clicking the **Start Recording** button, the path of selected green points will be traced and visualized to track exercise form throughout the duration of the recording

- **Count Exercise Repetitions**
    - Pick an exercise to count repetitions for from the dropdown selection
    - After clicking **Start Recording**, the repetitions will start count for the duration of the exercise recording
    - Changing exercise choice will reset the count, or you can opt for no counting with the **No Count** option in the dropdown

- **Download Recording**
    - After starting your recording and then stopping the recording, the video is then available to download
    - Clicking the download button beside the record button will initiate the download as a *webm* file
    - This button will normally stay blocked unless there is a video available to download


<!-- 
- **Upload Video**
    - TODO -->


## Getting Started

Clone the repo and once in the project directory run:

```
npm install
```
To install libraries and dependencies. This will install the TensorFlow Model, React Webcam, and some other libraries used to make the UI.

You can then run any of the following :

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.


