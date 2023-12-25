import './App.css'; 
import React, { useState, useEffect, useRef } from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import Webcam from 'react-webcam';
import { FaCloudUploadAlt } from 'react-icons/fa';

function App() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [poseDetector, setPoseDetector] = useState<poseDetection.PoseDetector | null>(null);

  useEffect(() => {
    const initializeTensorFlow = async () => {
      await tf.ready();
      const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING };
      const detectorInstance = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        detectorConfig
      );
      setPoseDetector(detectorInstance);
    };

    initializeTensorFlow();
  }, []);

  const detect = async () => {
    if (webcamRef.current && webcamRef.current.video && webcamRef.current.video.readyState === 4) {
      const video = webcamRef.current.video;
      if (poseDetector) {
        const poses = await poseDetector.estimatePoses(video);
        console.log(poses);

        drawPoses(poses, video.width, video.height);
      }
    }
  };

  const drawPoses = (poses: poseDetection.Pose[], videoWidth: number, videoHeight: number) => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        // Loop through each detected pose and draw it on the canvas
        poses.forEach((pose) => {
          pose.keypoints.forEach((keypoint) => {
            const { x, y } = keypoint;
            const circleColor = 'red';
            ctx.fillStyle = circleColor;
            ctx.beginPath();
            ctx.arc(x, y, 10, 0, 2 * Math.PI);
            ctx.fill();
          });
        });
      }
    }
  };

  // useEffect(() => {
  //   const detectionInterval = setInterval(() => {
  //     detect();
  //   }, 500); // Run every 0.5 seconds

  //   return () => clearInterval(detectionInterval); // Cleanup interval on component unmount
  // }, [detect]);

  return (
    <div className="min-h-screen bg-zinc-800 flex justify-center items-center">
      <div className="min-h-screen bg-zinc-800 flex flex-col justify-around">
        <div className="text-center">
          <h1 className="text-white text-6xl font-bold">Track My Form</h1>
        </div>
        <div className="m-0 p-0 relative text-center">
          <Webcam ref={webcamRef} className="z-5 m-0 p-0 w-[640px] h-[480px] border-dashed border-2 border-sky-50" />
          <canvas ref={canvasRef} className="bg-black z-10 absolute top-0 left-0 m-0 p-0 w-[640px] h-[480px]" />
        </div>
        <div className="flex flex-row justify-around">
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700" >
            RECORD
          </button>
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700" >
            DOWNLOAD
          </button>
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700" >
            SCREENSHOT
          </button>
          <label htmlFor="file-input" className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700">
            UPLOAD FILE
          </label>
          <input id="file-input" type="file" style={{ display: 'none' }}  />
        </div>
      </div>
    </div>
  );
}

export default App;
