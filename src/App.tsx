import './App.css'; 
import React, { useState, useEffect, useRef } from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import Webcam from 'react-webcam';
import { FaCloudUploadAlt } from 'react-icons/fa';
import { BsFillRecordFill } from "react-icons/bs";
import { IoMdDownload } from "react-icons/io";
import { RiScreenshot2Line } from "react-icons/ri";


function App() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [poseDetector, setPoseDetector] = useState<poseDetection.PoseDetector | null>(null);

  // Initialize TensorFlowJS and the MoveNet Model
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

  // Make predictions on webcam when ready
  const detect = async () => {
    if (webcamRef.current && webcamRef.current.video && webcamRef.current.video.readyState === 4) {
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      if (poseDetector) {
        // Make prediction and get position labels 
        const poses = await poseDetector.estimatePoses(video);

        drawPoses(poses, videoWidth, videoHeight);
      }
    }
  };

  // Draw Pose predictions on canvas
  const drawPoses = (poses: poseDetection.Pose[], videoWidth: number, videoHeight: number) => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        // adjust canvas dimensions to match video dimensions
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        //clear any previous drawings on canvas
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        // Loop through each detected pose and draw it on the canvas
        poses.forEach((pose) => {
          pose.keypoints.forEach((keypoint) => {
            const { x, y } = keypoint;
            const circleColor = 'red';
            ctx.fillStyle = circleColor;
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, 2 * Math.PI);
            ctx.fill();
          });
        });
      }
    }
  };

  // Prediction Loop
  useEffect(() => {
    const detectionInterval = setInterval(() => {
      detect();
    }, 100); // Run every 0.1 seconds

    return () => clearInterval(detectionInterval); // Cleanup interval on component unmount
  }, [detect]);

  return (
    <div className="min-h-screen bg-zinc-800 flex justify-center items-center">
      <div className="min-h-screen bg-zinc-800 flex flex-col justify-around">
        <div className="text-center">
          <h1 className="text-white text-6xl font-bold">Track My Form</h1>
        </div>
        <div className="m-0 p-0 relative text-center">
          <Webcam ref={webcamRef} className="z-5 m-0 p-0 w-[640px] h-[480px] border-dashed border-4 border-sky-50" />
          <canvas ref={canvasRef} className="z-10 absolute top-0 left-0 m-0 p-0 w-[640px] h-[480px] border-dashed border-4 border-sky-50" />
        </div>
        <div className="flex flex-row justify-around">
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700" >
            RECORD
            <BsFillRecordFill size={20}/>
          </button>
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700" >
            DOWNLOAD
            <IoMdDownload size={20}/>
          </button>
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700" >
            <RiScreenshot2Line size={20}/>
            SCREENSHOT
          </button>
          <label htmlFor="file-input" className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700">
           <FaCloudUploadAlt size={25}></FaCloudUploadAlt>
            UPLOAD FILE
          </label>
          <input id="file-input" type="file" style={{ display: 'none' }}  />
        </div>
      </div>
    </div>
  );
}

export default App;
