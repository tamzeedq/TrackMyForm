import './App.css'; 
import React, { useState, useEffect, useRef } from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import Webcam from 'react-webcam';
import { FaCloudUploadAlt, FaRobot, FaCamera, FaStop } from 'react-icons/fa';
import { BsFillRecordFill } from "react-icons/bs";
import { IoMdDownload } from "react-icons/io";
import { RiScreenshot2Line } from "react-icons/ri";
import * as params from './params';


function App() {
  const webcamRef = useRef<Webcam>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [poseDetector, setPoseDetector] = useState<poseDetection.PoseDetector | null>(null);
  const [lastDetectedPose, setLastDetectedPose] = useState<poseDetection.Pose | null>(null);
  const [showDetection, setShowDetection] = useState<boolean>(false);
  const [showWebcam, setShowWebCam] = useState<boolean>(true);
  const [recordingStatus, setRecordingStatus] = useState<boolean>(false);
  const [recordedVideo, setRecordedVideo] = useState([]);
  const [selectedKeypoints, setSelectedKepoints] = useState<string[]>([]);

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

      if (poseDetector && showDetection) {
        // Make prediction and get position labels 
        const poses = await poseDetector.estimatePoses(video);

        drawPoses(poses, videoWidth, videoHeight);
        setLastDetectedPose(poses[0]);

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
          
          drawPoseSkeleton(ctx, pose);
          drawPoseKeypoints(ctx, pose);

          if (recordingStatus && selectedKeypoints.length > 0) {
            traceSelectedPoints(ctx, pose);
          }
        });
      }
    }
  };

  const drawPoseSkeleton = (ctx : CanvasRenderingContext2D, pose: poseDetection.Pose) => {
    // Styles
    ctx.fillStyle = "white";
    ctx.strokeStyle = "white";
    ctx.lineWidth = params.DEFAULT_LINE_WIDTH;
    
    const keypoints = pose["keypoints"];
    
    // Get the adjacent points and draw a line connecting them
    poseDetection.util
                .getAdjacentPairs(poseDetection.SupportedModels.MoveNet)
                .forEach(([ i, j ]) => {
                
      // Get adjacent keypoints
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];
                  
      // TODO: improve score logic
      // If score is null, just show the keypoint.
      // const score1 = kp1.score != null ? kp1.score : 1;
      // const score2 = kp2.score != null ? kp2.score : 1;
      // const scoreThreshold = 0.3;

      // if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
      ctx.beginPath();
      ctx.moveTo(kp1.x, kp1.y);
      ctx.lineTo(kp2.x, kp2.y);
      ctx.stroke();
      // }
    });
  }

  // Trace path that keypoints of interest take
  const traceSelectedPoints = (ctx : CanvasRenderingContext2D, pose: poseDetection.Pose) => {
    ctx.fillStyle = "cornflowerblue";
    ctx.strokeStyle = "cornflowerblue";
    ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    // Draw line connecting current detection and previous detection
    pose.keypoints.forEach((keypoint, index) => {

      // If keypoint is of interest and there was a previous detection
      if (lastDetectedPose && 
        keypoint.name != undefined && 
        selectedKeypoints.includes(keypoint.name)) {

        ctx.beginPath();
        ctx.moveTo(lastDetectedPose.keypoints[index].x, lastDetectedPose.keypoints[index].y);
        ctx.lineTo(keypoint.x, keypoint.y);
        ctx.stroke();
      }

    });
  }

  const drawPoseKeypoints = (ctx : CanvasRenderingContext2D, pose: poseDetection.Pose) => {
    let circleColor : string;

    // Draw keypoints 
    pose.keypoints.forEach((keypoint) => {
      const { x, y } = keypoint;

      // console.log(selectedKeypoints, keypoint.name)
      if (keypoint.name && selectedKeypoints.includes(keypoint.name)) {
        circleColor = 'lime';
      } else {
        circleColor = 'red';
      }

      // Draw keypoint
      ctx.fillStyle = circleColor;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
      
    });
  }

  // Toggle Pose Detection
  const toggleDetection = () => {
    if (showDetection && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        //clear any previous drawings on canvas
        ctx.clearRect(0, 0, 640, 480);  
      }
    }

    setShowDetection(!showDetection)
  }

  // Highlight the closest detection to click
  const highlightClosestDetection = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;

    if (lastDetectedPose && canvas) {
      // Initialize comparison vals
      const canvasRect = canvas.getBoundingClientRect();
      const clickX = e.clientX - canvasRect.left;
      const clickY = e.clientY - canvasRect.top;  
      
      let closestKeypoint: string | null | undefined = null;
      let minDistance = Number.MAX_SAFE_INTEGER;

      lastDetectedPose.keypoints.forEach((keypoint) => {
        const distance = Math.sqrt((clickX - keypoint.x)**2 + (clickY - keypoint.y)**2);

        // Founds new min and click is within 25 pixels
        if(distance < minDistance && distance <= 30) {
          // Update closest vals
          closestKeypoint = keypoint.name;
          minDistance = distance;
        }
      })

      if (closestKeypoint) { // closest keypoint was found
        if (selectedKeypoints.includes(closestKeypoint)) {
          // remove clicked point from selected keypoints
          setSelectedKepoints(selectedKeypoints.filter((kp) => {return kp !== closestKeypoint}));
        } else { // add clicked point to selected keypoints
          setSelectedKepoints(selectedKeypoints.concat(closestKeypoint));
        }
      }
    }
  }
  

  // Camera Screenshot
  const screenshot = () => {
    if(webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      
      if (imageSrc) {
        // Create a link element
        const downloadLink = document.createElement('a');
        downloadLink.href = imageSrc;

        // Set the filename for the downloaded image
        const filename = 'screenshot.jpeg';
        downloadLink.download = filename;

        // Append the link to the body and trigger a click event
        document.body.appendChild(downloadLink);
        downloadLink.click();

        // Remove the link from the body
        document.body.removeChild(downloadLink);
      }
    }
  }

  // Camera Record 
  // Handle camera record click
  const handleRecordClick = () => {
    console.log("inside record click");
    if(!recordingStatus) {
      record();
    } else {
      console.log("stopping recording");
      stopRecording();
    }
  }

  // Append new recored data to what's already recorded
  const handleDataAvailable = (data:any) => {
    console.log(data);
    if (data) {
      console.log("got data");
      setRecordedVideo((prev) => prev.concat(data));
    }
  }

  // Make new recorder and start recording
  const record = () => {
    if (webcamRef.current && webcamRef.current.stream) {
      setRecordingStatus(true);
      mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
        mimeType: "video/webm"
      });
      mediaRecorderRef.current.addEventListener(
        "dataavailable",
        handleDataAvailable
      );
      mediaRecorderRef.current.start();
      console.log("started recording");
    }
  }

  // Download recorded video
  const downloadRecording = () => {
    if (recordedVideo.length > 0) {
      console.log("downloading");
      const blob = new Blob(recordedVideo, {
        type: "video/webm"
      });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement("a");
      document.body.appendChild(a);

      // Set attributes using setAttribute
      a.setAttribute("style", "display: none");
      a.href = url;
      a.download = "exercise-webcam.webm";
      a.click();

      window.URL.revokeObjectURL(url);
      setRecordedVideo([]);
    }
  };


  // Stop recording and update recording status
  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecordingStatus(false);
    }
  }

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
        <div className='flex flex-row justify-around items-center'>
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700"
                  onClick={toggleDetection} >
            <FaRobot  size={20}/>
            Toggle AI / Clear
          </button>
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700"
                  onClick={() => {setShowWebCam(!showWebcam)}} >
            <FaCamera  size={20}/>
            Toggle Web Cam
          </button>
        </div>
        <div className="m-0 p-0 relative text-center">
          <Webcam ref={webcamRef} 
            className="z-5 m-0 p-0 w-[640px] h-[480px] border-dashed border-4 border-sky-50" 
            screenshotFormat='image/jpeg'/>
          <canvas ref={canvasRef} 
            className={`z-10 absolute top-0 left-0 m-0 p-0 w-[640px] h-[480px] border-dashed border-4 
                      ${showWebcam ? '' : 'bg-black'}
                      ${recordingStatus ? 'border-red-400' : 'border-sky-50'}`} 
            onClick={highlightClosestDetection}/>
        </div>
        <div className="flex flex-row justify-around">
          <button className={`btn ${recordingStatus ? 'text-red-400' : 'text-white'}  bg-zinc-500 outline-none font-bold hover:bg-zinc-700`} 
                  onClick={handleRecordClick}>
            {recordingStatus ? (
              <>
                STOP RECORDING
                <FaStop  size={20} />
              </>
            ) : (
              <>
                START RECORDING
                <BsFillRecordFill size={20} />
              </>
            )}
          </button>
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700" 
                  onClick={downloadRecording}>
            DOWNLOAD
            <IoMdDownload size={20}/>
          </button>
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700"
                  onClick={screenshot} >
            <RiScreenshot2Line size={20}/>
            SCREENSHOT
          </button>
          <label htmlFor="file-input" className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700">
           <FaCloudUploadAlt size={25}/>
            UPLOAD FILE
          </label>
          <input id="file-input" type="file" style={{ display: 'none' }}  />
        </div>
      </div>
    </div>
  );
}

export default App;
