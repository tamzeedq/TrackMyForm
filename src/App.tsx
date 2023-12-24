import React, {useState, useEffect, useRef} from 'react';
import logo from './logo.svg';
import './App.css';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import Webcam from 'react-webcam';

function App() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [poseDetector, setPoseDetector] = useState<poseDetection.PoseDetector | null>(null);

  useEffect(() => {
    const initializeTensorFlow = async () => {
      await tf.ready();
      const detectorInstance = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
      );
      setPoseDetector(detectorInstance);
    };

    initializeTensorFlow();
  }, []);

  const detectorConfig = {modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING};
  const detector = async() => {
    const poseModel = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
    return poseModel;
  };

  const detect = async () => {
    if (webcamRef.current && webcamRef.current.video && webcamRef.current.video.readyState === 4) {
      const video = webcamRef.current.video;
      if (poseDetector) {
        const poses = await poseDetector.estimatePoses(video);
        console.log(poses); // Handle the detected poses as needed
      }
    }
  }

  useEffect(() => {
    const detectionInterval = setInterval(() => {
      detect();
    }, 500); // Run every 0.5 seconds

    return () => clearInterval(detectionInterval); // Cleanup interval on component unmount
  }, [detect]);

  return (
    <div className="App">
      <header className="App-header">
        {/* <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
        <a
          className="App-link" 
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a> */}
        {/* <p>hello world</p> */}
        <Webcam ref={webcamRef} className='absolute mx-auto z-9 text-center w-640 h-480'></Webcam>
        <canvas ref={canvasRef} className='absolute mx-auto text-center w-640 h-480'></canvas>
      </header>
    </div>
  );
}

export default App;
