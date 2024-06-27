// File: App.js
import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);
  const [isLiveMode, setIsLiveMode] = useState(true);
  const [detections, setDetections] = useState([]);
  const animationRef = useRef(null);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [objectCounts, setObjectCounts] = useState({});
  const [fps, setFps] = useState(0);
  const lastDetectionTime = useRef(Date.now());
  const frameCount = useRef(0);
  const [selectedObject, setSelectedObject] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [customClasses, setCustomClasses] = useState([]);
  const [newClassName, setNewClassName] = useState('');
  const [trainingImages, setTrainingImages] = useState({});

  const loadModel = useCallback(async () => {
    try {
      const loadedModel = await cocoSsd.load({base: 'mobilenet_v2'});
      setModel(loadedModel);
    } catch (err) {
      console.error("Error loading the model", err);
      setError("Failed to load the AI model. Please try refreshing the page.");
    }
  }, []);

  useEffect(() => {
    loadModel();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [loadModel]);

  const setupCamera = useCallback(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const constraints = { 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      };
      navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
          let video = videoRef.current;
          video.srcObject = stream;
          video.addEventListener('loadedmetadata', () => {
            video.play().then(() => setIsVideoPlaying(true)).catch(e => {
              console.error("Error playing the video:", e);
              setError("Failed to start the video stream. Please ensure you've granted camera permissions and try again.");
            });
          });
        })
        .catch((err) => {
          console.error("Error accessing the camera", err);
          setError("Failed to access the camera. Please ensure you've granted camera permissions and try again.");
        });
    }
  }, []);

  useEffect(() => {
    setupCamera();
  }, [setupCamera]);

  const detectObjects = useCallback(async (imageSource = null) => {
    if (model && (videoRef.current || imageSource) && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      let source = imageSource || videoRef.current;
      canvas.width = source.videoWidth || source.width;
      canvas.height = source.videoHeight || source.height;

      ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

      const predictions = await model.detect(canvas);

      ctx.font = '16px Arial';
      ctx.lineWidth = 2;

      const newObjectCounts = {};
      predictions.forEach((prediction, index) => {
        const [x, y, width, height] = prediction.bbox;
        const color = `hsl(${(index * 137) % 360}, 70%, 50%)`;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;

        ctx.strokeRect(x, y, width, height);
        ctx.fillText(
          `${prediction.class} (${Math.round(prediction.score * 100)}%)`,
          x, y > 20 ? y - 5 : 20
        );

        newObjectCounts[prediction.class] = (newObjectCounts[prediction.class] || 0) + 1;
      });

      // Add custom class detections
      for (let className of customClasses) {
        if (trainingImages[className] && trainingImages[className].length > 0) {
          const customDetection = await detectCustomClass(canvas, className);
          if (customDetection) {
            predictions.push(customDetection);
            newObjectCounts[className] = (newObjectCounts[className] || 0) + 1;

            const [x, y, width, height] = customDetection.bbox;
            ctx.strokeStyle = 'yellow';
            ctx.fillStyle = 'yellow';
            ctx.strokeRect(x, y, width, height);
            ctx.fillText(
              `${className} (${Math.round(customDetection.score * 100)}%)`,
              x, y > 20 ? y - 5 : 20
            );
          }
        }
      }

      setObjectCounts(newObjectCounts);
      setDetections(predictions);

      if (isLiveMode) {
        const now = Date.now();
        const elapsed = now - lastDetectionTime.current;
        frameCount.current++;
        if (elapsed > 1000) {
          setFps(Math.round((frameCount.current * 1000) / elapsed));
          frameCount.current = 0;
          lastDetectionTime.current = now;
        }

        animationRef.current = requestAnimationFrame(() => detectObjects());
      }
    }
  }, [model, isLiveMode, customClasses, trainingImages]);

  const detectCustomClass = async (image, className) => {
    const examples = trainingImages[className];
    if (!examples || examples.length === 0) return null;

    let maxSimilarity = -Infinity;
    let bestMatch = null;

    for (let example of examples) {
      const similarity = await compareImages(image, example);
      if (similarity > maxSimilarity) {
        maxSimilarity = similarity;
        bestMatch = example;
      }
    }

    if (maxSimilarity > 0.7) { // Adjust this threshold as needed
      return {
        class: className,
        score: maxSimilarity,
        bbox: [0, 0, image.width, image.height] // Full image bounding box
      };
    }

    return null;
  };

  const compareImages = async (img1, img2) => {
    const tensor1 = tf.browser.fromPixels(img1);
    const tensor2 = tf.browser.fromPixels(img2);

    const resized1 = tf.image.resizeBilinear(tensor1, [224, 224]);
    const resized2 = tf.image.resizeBilinear(tensor2, [224, 224]);

    const normalized1 = resized1.div(255);
    const normalized2 = resized2.div(255);

    const similarity = tf.metrics.cosineProximity(normalized1.flatten(), normalized2.flatten()).dataSync()[0];

    tensor1.dispose();
    tensor2.dispose();
    resized1.dispose();
    resized2.dispose();
    normalized1.dispose();
    normalized2.dispose();

    return similarity;
  };

  useEffect(() => {
    if (isLiveMode && model && isVideoPlaying) {
      detectObjects();
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isLiveMode, model, detectObjects, isVideoPlaying]);

  const handleModeChange = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    setIsLiveMode(!isLiveMode);
    if (isLiveMode) {
      setDetections([]);
      setObjectCounts({});
      setCapturedImage(null);
    } else {
      detectObjects();
    }
  };

  const handleDetectAndSave = () => {
    if (!isLiveMode && videoRef.current) {
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      const imageDataUrl = canvas.toDataURL('image/jpeg');
      setCapturedImage(imageDataUrl);

      const img = new Image();
      img.onload = () => detectObjects(img);
      img.src = imageDataUrl;
    }
  };

  const handleObjectSelect = (objectClass) => {
    setSelectedObject(objectClass);
  };

  const handleAddClass = () => {
    if (newClassName && !customClasses.includes(newClassName)) {
      setCustomClasses([...customClasses, newClassName]);
      setNewClassName('');
    }
  };

  const handleImageUpload = (e, className) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setTrainingImages(prev => ({
            ...prev,
            [className]: [...(prev[className] || []), img]
          }));
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="App">
      <h1>AI Object Detection with Custom Training</h1>
      <div style={{ position: 'relative' }}>
        <video
          ref={videoRef}
          style={{ width: '100%', maxWidth: '640px', height: 'auto' }}
          autoPlay
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            maxWidth: '640px',
            height: 'auto'
          }}
        />
      </div>
      <div>
        <button onClick={handleModeChange}>
          {isLiveMode ? "Switch to Detect and Save" : "Switch to Live Mode"}
        </button>
        {!isLiveMode && (
          <button onClick={handleDetectAndSave}>Detect and Save</button>
        )}
      </div>
      {capturedImage && !isLiveMode && (
        <div>
          <h3>Captured Image:</h3>
          <img src={capturedImage} alt="Captured" style={{ maxWidth: '100%', height: 'auto' }} />
        </div>
      )}
      <div>
        <h2>Detection Stats:</h2>
        {isLiveMode && <p>FPS: {fps}</p>}
        <h3>Object Counts:</h3>
        <ul>
          {Object.entries(objectCounts).map(([objectClass, count]) => (
            <li key={objectClass} onClick={() => handleObjectSelect(objectClass)} style={{cursor: 'pointer'}}>
              {objectClass}: {count}
            </li>
          ))}
        </ul>
      </div>
      {selectedObject && (
        <div>
          <h3>Details for {selectedObject}:</h3>
          <ul>
            {detections
              .filter(detection => detection.class === selectedObject)
              .map((detection, index) => (
                <li key={index}>
                  Confidence: {Math.round(detection.score * 100)}%,
                  Position: (x: {Math.round(detection.bbox[0])}, y: {Math.round(detection.bbox[1])}),
                  Size: {Math.round(detection.bbox[2])}x{Math.round(detection.bbox[3])}
                </li>
              ))}
          </ul>
        </div>
      )}
      <div>
        <h2>Custom Training</h2>
        <input
          type="text"
          value={newClassName}
          onChange={(e) => setNewClassName(e.target.value)}
          placeholder="Enter new class name"
        />
        <button onClick={handleAddClass}>Add Class</button>
        
        {customClasses.map(className => (
          <div key={className}>
            <h3>{className}</h3>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => handleImageUpload(e, className)}
            />
            <p>Training images: {trainingImages[className]?.length || 0}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;