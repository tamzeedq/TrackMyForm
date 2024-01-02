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

  // Prediction Loop
  useEffect(() => {
    const detectionInterval = setInterval(() => {
      detect();
    }, 100); // Run every x seconds

    return () => clearInterval(detectionInterval); // Cleanup interval on component unmount
  }, [detect]);

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

  // -------------- Drawing --------------
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

        // Redraw traces
        if (recordingStatus && selectedKeypoints.length > 0 && traceImage) {
          ctx.putImageData(traceImage, 0, 0);
        }

        // Loop through each detected pose and draw it on the canvas
        poses.forEach((pose) => {

          // If recording and points are selected
          if (recordingStatus && selectedKeypoints.length > 0) {
            traceSelectedPoints(ctx, pose);
            setTraceImage(ctx.getImageData(0, 0, 640, 480)); // save traces
          }

          // Draw kepoints and skeleton of pose
          drawPoseSkeleton(ctx, pose);
          drawPoseKeypoints(ctx, pose);

        });
      }
    }
  };

  // Draw the lines that connect keypoints
  const drawPoseSkeleton = (ctx : CanvasRenderingContext2D, pose: poseDetection.Pose) => {
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
  }

  // Draw the keypoints of the pose (joints, nose, eyes, etc)
  const drawPoseKeypoints = (ctx : CanvasRenderingContext2D, pose: poseDetection.Pose) => {

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
  }

  // Trace path that keypoints of interest take
  const traceSelectedPoints = (ctx : CanvasRenderingContext2D, pose: poseDetection.Pose) => {
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
  }

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
  
  // -------------- Exercise UI --------------
  // Handle picking an exercise to count reps from drop down
  const chooseDropdownExercise = (e : React.MouseEvent<HTMLAnchorElement, MouseEvent>, exercise:string | null) => {
    e.preventDefault(); 
    setCurrExercise(exercise); 
    setRepCount(0);
    setHalfwayStatus(false);
  }

  // Handle checking Push-Up Form
  const checkPushUpForm = (pose : poseDetection.Pose) => {
    if (pose.keypoints) {

      // if left and right shoulder are below left and right elbow
      if ((pose.keypoints[6].y > pose.keypoints[8].y) || 
          (pose.keypoints[5].y > pose.keypoints[7].y)) {
        
        setHalfwayStatus(true); // Concentric was completed

      } else if ((pose.keypoints[6].y < pose.keypoints[8].y) && 
                (pose.keypoints[5].y < pose.keypoints[7].y) && halfwayStatus) {
        // if left and right shoulder are above left and right elbow and essentric was completed

        setRepCount((prev) => prev + 1); // increment rep count
        setHalfwayStatus(false); // Concentric was completed, rep complete

      }
    }
  }

  // Handle checking Pull-Up Form
  const checkPullUpForm = (pose : poseDetection.Pose) => {
    
    if (pose.keypoints) {
      // if left and right shoulder are below left and right elbow
      if ((pose.keypoints[6].y > pose.keypoints[8].y) && 
          (pose.keypoints[5].y > pose.keypoints[7].y)) {

        setHalfwayStatus(true); // Essentric was completed

      } else if ((pose.keypoints[6].y <= pose.keypoints[8].y) && 
                (pose.keypoints[5].y <= pose.keypoints[7].y) && halfwayStatus) {
        // if left and right shoulder are above left and right elbow

        setRepCount((prev) => prev + 1); // increment rep count
        setHalfwayStatus(false); // Concentric was completed, rep complete
      }
    }
  }

  // Handle checking Squat Form
  const checkSquatForm = (pose : poseDetection.Pose) => {
    if (pose.keypoints) {

      // if left and right hip are below left and right knee
      if ((pose.keypoints[12].y >= pose.keypoints[14].y) || 
          (pose.keypoints[11].y >= pose.keypoints[13].y)) {
    
        setHalfwayStatus(true); // Essentric was completed

      } else if ((pose.keypoints[11].y < pose.keypoints[13].y) && 
                (pose.keypoints[12].y < pose.keypoints[14].y) && halfwayStatus) {
        // if left and right hip are above left and right knee and essentric was completed
        
        setRepCount((prev) => prev + 1); // increment rep count
        setHalfwayStatus(false); // Concentric was completed, rep complete

      }
    }
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
  
  // Handle camera record click
  const handleStartCaptureClick = React.useCallback(() => {
    if (webcamRef.current && webcamRef.current.stream) {
      setRecordingStatus(true);
      setRecordedVideo([]);

      // Make new recorder and start recording
      mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
        mimeType: "video/webm"
      });
      mediaRecorderRef.current.addEventListener(
        "dataavailable",
        handleDataAvailable
      );

      mediaRecorderRef.current.start();
    }
  }, [webcamRef, setRecordingStatus, mediaRecorderRef]);

  // Concatenate recorded video
  const handleDataAvailable = React.useCallback(
    ({ data }: { data: any }) => {
      if (data.size > 0) {
        setRecordedVideo((prev) => prev.concat(data));
      }
    },
    [setRecordedVideo]
  );

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
  }, [mediaRecorderRef, webcamRef, setRecordingStatus, setHalfwayStatus]);

  return (
    <div className="min-h-screen bg-zinc-800 flex justify-center items-center">
      <div className="min-h-screen bg-zinc-800 flex flex-col justify-around">
        <div className="text-center">
          <h1 className="text-white text-6xl font-bold">Track My Form</h1>
        </div>
        <div className='flex flex-row justify-around items-center'>
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700"
                  onClick={toggleDetection} >
            <FaRobot size={20}/>
            Toggle AI / Clear
          </button>
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700"
                  onClick={() => {setShowWebCam(!showWebcam)}} >
            <FaCamera size={20}/>
            Toggle Webcam
          </button>
          <div className="dropdown">
            <div tabIndex={0} role="button" className="btn bg-zinc-500 hover:bg-zinc-700 text-white m-1"> <PiPersonArmsSpread size={20}/> {currExercise ? currExercise : <>Pick Exercise</>}</div>
            <ul tabIndex={0} className="dropdown-content bg-zinc-500 text-white z-[20] menu p-2 shadow bg-base-100 rounded-box w-52">
              <li><a href='/' className='hover:bg-zinc-700' onClick={(e) => { chooseDropdownExercise(e, null)}}>No Count</a></li>
              <li><a href='/' className='hover:bg-zinc-700' onClick={(e) => { chooseDropdownExercise(e,"Push-Up")}}>Push-Up</a></li>
              <li><a href='/' className='hover:bg-zinc-700' onClick={(e) => { chooseDropdownExercise(e,"Pull-Up")}}>Pull-Up</a></li>
              <li><a href='/' className='hover:bg-zinc-700' onClick={(e) => { chooseDropdownExercise(e,"Squat")}}>Squat</a></li>
            </ul>
          </div>
          { currExercise ? <div className='text-white text-xl'>Reps: {repCount}</div>: <></>}
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
          {recordingStatus ? 
            <button className="btn text-red-400 bg-zinc-500 outline-none font-bold hover:bg-zinc-700" 
              onClick={handleStopCaptureClick}>
              STOP RECORDING
              <FaStop  size={20} />
            </button>
            :
            <button className="btn text-white bg-zinc-500 outline-none font-bold hover:bg-zinc-700" 
              onClick={handleStartCaptureClick}>
              START RECORDING
              <BsFillRecordFill size={20} />
            </button>
          }
          <button className={`btn ${(recordedVideo.length > 0) ? '' : 'btn-disabled'} bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700`} 
                  onClick={downloadRecording}>
            DOWNLOAD
            <IoMdDownload size={20}/>
          </button>
          <button className="btn bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700"
                  onClick={screenshot} >
            <RiScreenshot2Line size={20}/>
            SCREENSHOT
          </button>
          <label htmlFor="file-input" className=" btn btn-disabled bg-zinc-500 outline-none text-white font-bold hover:bg-zinc-700">
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
