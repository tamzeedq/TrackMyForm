import './App.css'; 
import React, { useState, useEffect, useRef } from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import Webcam from 'react-webcam';
import { FaRobot, FaCamera, FaStop, FaGithub } from 'react-icons/fa';
import { BsFillRecordFill } from "react-icons/bs";
import { IoMdDownload } from "react-icons/io";
import { RiScreenshot2Line } from "react-icons/ri";
import { PiPersonArmsSpread } from "react-icons/pi";
import * as params from './params';


function App() {
  const webcamRef = useRef<Webcam>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // State variables for Pose Detection
  const [poseDetector, setPoseDetector] = useState<poseDetection.PoseDetector | null>(null);
  const [lastDetectedPose, setLastDetectedPose] = useState<poseDetection.Pose | null>(null);
  const [selectedKeypoints, setSelectedKepoints] = useState<string[]>([]);
  const [traceImage, setTraceImage] = useState<ImageData | null>(null);

  // State variables for UI
  const [showDetection, setShowDetection] = useState<boolean>(false);
  const [showWebcam, setShowWebCam] = useState<boolean>(true);
  const [recordingStatus, setRecordingStatus] = useState<boolean>(false);
  const [recordedVideo, setRecordedVideo] = useState([]);

  // State variables to monitor exercise
  const [currExercise, setCurrExercise] = useState<string | null>();
  const [repCount, setRepCount] = useState<number>(0);
  // check if halfway point of rep has been reached
  const [halfwayStatus, setHalfwayStatus] = useState<boolean>(false);

  // Mobile friendly
  const webcamDimensions = { width: 640, height: 480 };
  const [isMobile, setIsMobile] = useState(false);
  const [videoConstraints, setVideoConstraints] = useState({
    width: 640,
    height: 480,
    facingMode: 'user'
  });

  // -------------- Mobile Friendly --------------
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
      setVideoConstraints(prev => ({
        ...prev,
        width: window.innerWidth < 768 ? window.innerWidth - 32 : 640,
        height: window.innerWidth < 768 ? (window.innerWidth - 32) * 0.75 : 480
      }));
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // -------------- Exercise UI --------------
  // Handle picking an exercise to count reps from drop down
  const chooseDropdownExercise = (e : React.MouseEvent<HTMLButtonElement, MouseEvent>, exercise:string | null) => {
    e.preventDefault(); 
    setCurrExercise(exercise); 
    setRepCount(0);
    setHalfwayStatus(false);
  }
  // Handle checking Push-Up Form
  const checkPushUpForm = React.useCallback((pose: poseDetection.Pose) => {
    if (pose.keypoints) {
      if ((pose.keypoints[6].y > pose.keypoints[8].y) || 
          (pose.keypoints[5].y > pose.keypoints[7].y)) {
        setHalfwayStatus(true);
      } else if ((pose.keypoints[6].y < pose.keypoints[8].y) && 
                (pose.keypoints[5].y < pose.keypoints[7].y) && halfwayStatus) {
        setRepCount((prev) => prev + 1);
        setHalfwayStatus(false);
      }
    }
  }, [halfwayStatus]);

  // Handle checking Pull-Up Form
  const checkPullUpForm = React.useCallback((pose: poseDetection.Pose) => {
    if (pose.keypoints) {
      if ((pose.keypoints[6].y > pose.keypoints[8].y) && 
          (pose.keypoints[5].y > pose.keypoints[7].y)) {
        setHalfwayStatus(true);
      } else if ((pose.keypoints[6].y <= pose.keypoints[8].y) && 
                (pose.keypoints[5].y <= pose.keypoints[7].y) && halfwayStatus) {
        setRepCount((prev) => prev + 1);
        setHalfwayStatus(false);
      }
    }
  }, [halfwayStatus]);

  // Handle checking Squat Form
  const checkSquatForm = React.useCallback((pose: poseDetection.Pose) => {
    if (pose.keypoints) {
      if ((pose.keypoints[12].y >= pose.keypoints[14].y) || 
          (pose.keypoints[11].y >= pose.keypoints[13].y)) {
        setHalfwayStatus(true);
      } else if ((pose.keypoints[11].y < pose.keypoints[13].y) && 
                (pose.keypoints[12].y < pose.keypoints[14].y) && halfwayStatus) {
        setRepCount((prev) => prev + 1);
        setHalfwayStatus(false);
      }
    }
  }, [halfwayStatus]);

  //  -------------- Pose Detection ----------------
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

  // -------------- Drawing --------------
  // Draw the lines that connect keypoints
  const drawPoseSkeleton = React.useCallback((ctx: CanvasRenderingContext2D, pose: poseDetection.Pose) => {
    // Styles
    ctx.fillStyle = params.DEFAULT_SKELETON_COLOR;
    ctx.strokeStyle = params.DEFAULT_SKELETON_COLOR;
    ctx.lineWidth = params.DEFAULT_LINE_WIDTH;
    
    const keypoints = pose["keypoints"];
    
    // Get the adjacent points and draw a line connecting them
    poseDetection.util
                .getAdjacentPairs(poseDetection.SupportedModels.MoveNet)
                .forEach(([ i, j ]) => {
                
      // Get adjacent keypoints
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];
                  
      // Check if keypoint confidence is high enough to draw
      const scoreThreshold = params.SKELETON_CONFIDENCE;
      const score1 = kp1.score && kp1.score >= scoreThreshold;
      const score2 = kp2.score && kp2.score >= scoreThreshold;

      if (score1 && score2) {
        ctx.beginPath();
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
        ctx.stroke();
      }
    });
  }, []);

  // Draw the keypoints of the pose (joints, nose, eyes, etc)
  const drawPoseKeypoints = React.useCallback((ctx: CanvasRenderingContext2D, pose: poseDetection.Pose) => {

    // Draw keypoints 
    pose.keypoints.forEach((keypoint) => {
      if (keypoint.score && keypoint.score < 0.2) {return};
      
      const { x, y } = keypoint;

      if (keypoint.name && selectedKeypoints.includes(keypoint.name)) {
        ctx.fillStyle = params.SELECTED_KEYPOINT_COLOR;
      } else {
        ctx.fillStyle = params.KEYPOINT_COLOR;
      }

      // Draw keypoint
      ctx.beginPath();
      ctx.arc(x, y, params.DEFAULT_RADIUS, 0, 2 * Math.PI);
      ctx.fill();
      
    });
  }, [selectedKeypoints]);

  // Trace path that keypoints of interest take
  const traceSelectedPoints = React.useCallback((ctx: CanvasRenderingContext2D, pose: poseDetection.Pose) => {
    ctx.fillStyle = params.DEFAULT_TRACE_COLOR;
    ctx.strokeStyle = params.DEFAULT_TRACE_COLOR;
    ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    // Draw line connecting current detection and previous detection
    pose.keypoints.forEach((keypoint, index) => {

      // If keypoint is of interest and there was a previous detection
      if (lastDetectedPose && 
        keypoint.name !== undefined && 
        selectedKeypoints.includes(keypoint.name)) {

        ctx.beginPath();
        ctx.moveTo(lastDetectedPose.keypoints[index].x, lastDetectedPose.keypoints[index].y);
        ctx.lineTo(keypoint.x, keypoint.y);
        ctx.stroke();
      }

    });
  }, [lastDetectedPose, selectedKeypoints]);

  // Highlight the closest detection to click on webcam
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

  // Draw Pose predictions on canvas
  const drawPoses = React.useCallback((poses: poseDetection.Pose[], videoWidth: number, videoHeight: number) => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  
        if (recordingStatus && selectedKeypoints.length > 0 && traceImage) {
          ctx.putImageData(traceImage, 0, 0);
        }
  
        poses.forEach((pose) => {
          if (recordingStatus && selectedKeypoints.length > 0) {
            traceSelectedPoints(ctx, pose);
            setTraceImage(ctx.getImageData(0, 0, 640, 480));
          }
          drawPoseSkeleton(ctx, pose);
          drawPoseKeypoints(ctx, pose);
        });
      }
    }
  }, [recordingStatus, selectedKeypoints, traceImage, traceSelectedPoints, drawPoseSkeleton, drawPoseKeypoints]);

  useEffect(() => {
    // Make predictions on webcam when ready
    const detect = async () => {
      if (webcamRef.current && webcamRef.current.video && webcamRef.current.video.readyState === 4) {
        const video = webcamRef.current.video;
        const videoWidth = webcamRef.current.video.videoWidth;
        const videoHeight = webcamRef.current.video.videoHeight;

        if (poseDetector && showDetection) {
          // Make prediction and get position labels 
          const poses = await poseDetector.estimatePoses(video);
          
          // If recording and excerise is selected
          if (recordingStatus && currExercise) {
            // Check rep progress based on exercise
            if (currExercise === "Push-Up") {checkPushUpForm(poses[0]);} 
            else if (currExercise === "Pull-Up") {checkPullUpForm(poses[0]);} 
            else if (currExercise === "Squat") {checkSquatForm(poses[0]);}
          }

          // Draw predictions on canvas
          drawPoses(poses, videoWidth, videoHeight);
          setLastDetectedPose(poses[0]);
        }
      }
    };

    const detectionInterval = setInterval(() => {
      detect();
    }, 100);

    return () => clearInterval(detectionInterval);
  }, [poseDetector, showDetection, recordingStatus, currExercise, checkPushUpForm, checkPullUpForm, checkSquatForm, drawPoses]);

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
  
  // -------------- Camera / Recording --------------
  // Original Example : https://codepen.io/mozmorris/pen/yLYKzyp?editors=0010

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

  // Concatenate recorded video
  const handleDataAvailable = React.useCallback(
    ({ data }: { data: any }) => {
      if (data.size > 0) {
        setRecordedVideo((prev) => prev.concat(data));
      }
    },
    [setRecordedVideo]
  );
  
  // Handle camera record click
  const handleStartCaptureClick = React.useCallback(() => {
    if (webcamRef.current && webcamRef.current.stream) {
      setRecordingStatus(true);
      setRecordedVideo([]);
      setTraceImage(null);
  
      mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
        mimeType: "video/webm"
      });
      mediaRecorderRef.current.addEventListener(
        "dataavailable",
        handleDataAvailable
      );
  
      mediaRecorderRef.current.start();
    }
  }, [handleDataAvailable]); // Add handleDataAvailable to dependencies

  // Download recorded video
  const downloadRecording = () => {
    if (recordedVideo.length > 0) {

      // Create a blob from the recorded video
      const blob = new Blob(recordedVideo, {
        type: "video/webm"
      });

      // Create a URL from the blob
      const url = URL.createObjectURL(blob);
      
      // Create an invisible download link element
      const a = document.createElement("a");
      document.body.appendChild(a);

      a.setAttribute("style", "display: none");
      a.href = url;
      a.download = "exercise_webcam.webm";
      a.click(); // Trigger a click on the element to start download

      window.URL.revokeObjectURL(url); // Release the URL object
      setRecordedVideo([]);
    }
  };

  // Stop recording and update recording status
  const handleStopCaptureClick = React.useCallback(() => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecordingStatus(false);
      setHalfwayStatus(false);
    }
  }, [mediaRecorderRef, setRecordingStatus, setHalfwayStatus]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-900 to-zinc-800 flex justify-center items-center p-4 sm:p-8">
    <div className="w-full max-w-6xl space-y-4 sm:space-y-8">
      {/* Header Section */}
      <div className="text-center space-y-2 sm:space-y-4">
        <h1 className="text-3xl sm:text-5xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text">
          Track My Form
        </h1>
        <p className="text-sm sm:text-base text-zinc-400">AI-powered exercise form tracking and analysis</p>
      </div>

      {/* Main Controls */}
      <div className="flex flex-wrap gap-2 sm:gap-4 justify-center items-center">
        <button 
          className="btn btn-sm sm:btn-lg bg-zinc-700 text-white hover:bg-zinc-600 transition-all duration-300 space-x-2"
          onClick={toggleDetection}
        >
          <FaRobot className="text-blue-400" />
          <span className="text-sm sm:text-base">Toggle AI</span>
        </button>

        <button 
          className="btn btn-sm sm:btn-lg bg-zinc-700 text-white hover:bg-zinc-600 transition-all duration-300 space-x-2"
          onClick={() => setShowWebCam(!showWebcam)}
        >
          <FaCamera className="text-blue-400" />
          <span className="text-sm sm:text-base">Toggle Camera</span>
        </button>

        <div className="dropdown z-10">
          <button className="btn btn-sm sm:btn-lg bg-zinc-700 text-white hover:bg-zinc-600 transition-all duration-300 space-x-2">
            <PiPersonArmsSpread className="text-blue-400" />
            <span className="text-sm sm:text-base">{currExercise || 'Select Exercise'}</span>
          </button>
          <ul className="dropdown-content menu p-2 shadow-lg bg-zinc-700 rounded-lg mt-2 w-40 sm:w-52">
            <li>
              <button onClick={(e) => chooseDropdownExercise(e, null)} 
                className="text-sm sm:text-base text-white hover:bg-zinc-600 p-2 sm:p-3 rounded transition-all">
                No Count
              </button>
            </li>
            {['Push-Up', 'Pull-Up', 'Squat'].map((exercise) => (
              <li key={exercise}>
                <button
                  onClick={(e) => chooseDropdownExercise(e, exercise)}
                  className="text-sm sm:text-base text-white hover:bg-zinc-600 p-2 sm:p-3 rounded transition-all"
                >
                  {exercise}
                </button>
              </li>
            ))}
          </ul>
        </div>

        {currExercise && (
          <div className="px-4 sm:px-6 py-2 sm:py-3 bg-zinc-700 rounded-lg">
            <div className="text-xl sm:text-2xl font-bold text-blue-400">{repCount}</div>
            <div className="text-xs sm:text-sm text-zinc-400">Reps</div>
          </div>
        )}
      </div>

      {/* Camera Section */}
      <div 
        className="relative rounded-lg overflow-hidden shadow-2xl mx-auto w-full"
        style={{
          maxWidth: isMobile ? '100%' : '640px',
          height: isMobile ? `${videoConstraints.height}px` : '480px'
        }}
      >
        <Webcam
          ref={webcamRef}
          className="w-full h-full object-cover"
          screenshotFormat="image/jpeg"
          videoConstraints={{
            width: webcamDimensions.width,
            height: webcamDimensions.height,
            facingMode: "user"
          }}
        />
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 w-full h-full 
            ${!showWebcam && 'bg-black'}
            ${recordingStatus ? 'border-2 border-red-400' : 'border-2 border-blue-400'} 
            rounded-lg transition-all duration-300`}
          onClick={highlightClosestDetection}
        />
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-2 sm:gap-4 justify-center">
        <button
          className={`btn btn-sm sm:btn-lg ${
            recordingStatus 
              ? 'bg-red-500 hover:bg-red-600' 
              : 'bg-zinc-700 hover:bg-zinc-600'
          } text-white transition-all duration-300 space-x-2`}
          onClick={recordingStatus ? handleStopCaptureClick : handleStartCaptureClick}
        >
          {recordingStatus ? (
            <>
              <FaStop className="text-white" />
              <span className="text-sm sm:text-base">Stop</span>
            </>
          ) : (
            <>
              <BsFillRecordFill className="text-red-500" />
              <span className="text-sm sm:text-base">Record</span>
            </>
          )}
        </button>

        <button
          className={`btn btn-sm sm:btn-lg bg-zinc-700 text-white transition-all duration-300 space-x-2 
            ${recordedVideo.length === 0 && 'opacity-50 cursor-not-allowed'}`}
          onClick={downloadRecording}
          disabled={recordedVideo.length === 0}
        >
          <IoMdDownload className="text-blue-400" />
          <span className="text-sm sm:text-base">Download</span>
        </button>

        <button
          className="btn btn-sm sm:btn-lg bg-zinc-700 text-white hover:bg-zinc-600 transition-all duration-300 space-x-2"
          onClick={screenshot}
        >
          <RiScreenshot2Line className="text-blue-400" />
          <span className="text-sm sm:text-base">Screenshot</span>
        </button>

        <a
          href="https://github.com/tamzeedq/TrackMyForm"
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-sm sm:btn-lg bg-zinc-700 text-white hover:bg-zinc-600 transition-all duration-300 space-x-2"
        >
          <FaGithub className="text-blue-400" />
          <span className="text-sm sm:text-base">GitHub</span>
        </a>
      </div>
    </div>
  </div>
  );
}

export default App;

