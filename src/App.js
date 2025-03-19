// File: App.js
import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('detect'); // 'detect', 'capture', 'train'
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
  const [currentTrainingClass, setCurrentTrainingClass] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.3); // Lower default threshold for better detection
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [customModel, setCustomModel] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [detectionHistory, setDetectionHistory] = useState([]);
  const [modelLoading, setModelLoading] = useState(true);

  // Load the pre-trained COCO-SSD model with better error handling
  const loadModel = useCallback(async () => {
    try {
      setModelLoading(true);
      setTrainingLogs(prev => [...prev, "Loading COCO-SSD model..."]);
      
      // Force TensorFlow.js to use WebGL backend for better performance
      await tf.setBackend('webgl');
      
      // Load the model with a more optimized configuration
      const loadedModel = await cocoSsd.load({
        base: 'mobilenet_v2',
        modelUrl: undefined, // Let it use the default URL
      });
      
      setModel(loadedModel);
      setModelLoading(false);
      setTrainingLogs(prev => [...prev, "COCO-SSD model loaded successfully!"]);
      
      // Initialize detection after model is loaded
      if (mode === 'detect') {
        setTimeout(() => {
          if (videoRef.current && isVideoPlaying) {
            detectObjects();
          }
        }, 1000); // Small delay to ensure everything is ready
      }
    } catch (err) {
      console.error("Error loading the model", err);
      setTrainingLogs(prev => [...prev, `Error loading model: ${err.message}`]);
      setError("Failed to load the AI model. Please try refreshing the page.");
      setModelLoading(false);
    }
  }, [mode, isVideoPlaying]); // Added dependencies to trigger re-detection

  useEffect(() => {
    loadModel();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [loadModel]);

  // Set up camera access with improved error handling and resolution
  const setupCamera = useCallback(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      // Try to get higher resolution for better detection
      const constraints = { 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 } // Specify frame rate for smoother video
        } 
      };
      
      navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
          let video = videoRef.current;
          if (video) {
            video.srcObject = stream;
            
            // Use loadeddata instead of loadedmetadata for more reliable video loading
            video.addEventListener('loadeddata', () => {
              video.play()
                .then(() => {
                  setIsVideoPlaying(true);
                  console.log("Video playing successfully");
                  
                  // Properly size the canvas to match the video
                  if (canvasRef.current && video.videoWidth) {
                    canvasRef.current.width = video.videoWidth;
                    canvasRef.current.height = video.videoHeight;
                    console.log(`Canvas sized to: ${video.videoWidth}x${video.videoHeight}`);
                  }
                  
                  // Start detection if model is already loaded
                  if (model && mode === 'detect') {
                    detectObjects();
                  }
                })
                .catch(e => {
                  console.error("Error playing the video:", e);
                  setError("Failed to start the video stream. Please ensure you've granted camera permissions and try again.");
                });
            });
          }
        })
        .catch((err) => {
          console.error("Error accessing the camera", err);
          setError("Failed to access the camera. Please ensure you've granted camera permissions and try again.");
          
          // Fallback to lower resolution if the initial request fails
          const fallbackConstraints = { 
            video: { 
              facingMode: 'environment',
              width: { ideal: 640 },
              height: { ideal: 480 }
            } 
          };
          
          navigator.mediaDevices.getUserMedia(fallbackConstraints)
            .then((fallbackStream) => {
              let video = videoRef.current;
              if (video) {
                video.srcObject = fallbackStream;
                video.addEventListener('loadeddata', () => {
                  video.play()
                    .then(() => {
                      setIsVideoPlaying(true);
                      console.log("Video playing with fallback resolution");
                      
                      if (canvasRef.current && video.videoWidth) {
                        canvasRef.current.width = video.videoWidth;
                        canvasRef.current.height = video.videoHeight;
                      }
                      
                      if (model && mode === 'detect') {
                        detectObjects();
                      }
                    })
                    .catch(e => {
                      console.error("Error playing the video with fallback:", e);
                      setError("Failed to start the video stream even with fallback settings.");
                    });
                });
              }
            })
            .catch(fallbackErr => {
              console.error("Error accessing the camera with fallback settings:", fallbackErr);
              setError("Failed to access the camera. Please try a different browser or device.");
            });
        });
    } else {
      setError("Your browser doesn't support camera access. Please try a different browser.");
    }
  }, [model, mode]);

  useEffect(() => {
    setupCamera();
  }, [setupCamera]);

  // Main object detection function using COCO-SSD - improved for reliability
  const detectObjects = useCallback(async (imageSource = null) => {
    if (!model) {
      console.log("Model not loaded yet, skipping detection");
      return;
    }
    
    if ((!videoRef.current && !imageSource) || !canvasRef.current) {
      console.log("Video or canvas not ready, skipping detection");
      return;
    }
    
    try {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      let source = imageSource || videoRef.current;
      
      // Check if source dimensions are valid
      const sourceWidth = source.videoWidth || source.width;
      const sourceHeight = source.videoHeight || source.height;
      
      if (!sourceWidth || !sourceHeight) {
        console.log("Source dimensions not ready yet, retrying detection in 500ms");
        setTimeout(() => detectObjects(), 500);
        return;
      }
      
      // Ensure canvas is properly sized
      if (canvas.width !== sourceWidth || canvas.height !== sourceHeight) {
        canvas.width = sourceWidth;
        canvas.height = sourceHeight;
        console.log(`Canvas resized to: ${sourceWidth}x${sourceHeight}`);
      }

      // Draw the current frame
      ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

      // Detect with COCO-SSD
      console.log("Running detection...");
      const predictions = await model.detect(canvas, { scoreThreshold: confidenceThreshold });
      console.log("Detection results:", predictions);

      // Clear previous drawings
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

      // Set up text drawing parameters
      ctx.font = 'bold 16px Arial';
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';

      const newObjectCounts = {};
      predictions.forEach((prediction, index) => {
        const [x, y, width, height] = prediction.bbox;
        const color = getColorForClass(prediction.class);
        
        // Draw box with thicker border
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.strokeRect(x, y, width, height);
        
        // Draw label with background for better visibility
        const textWidth = ctx.measureText(`${prediction.class} (${Math.round(prediction.score * 100)}%)`).width;
        ctx.fillStyle = color;
        ctx.fillRect(x, y > 25 ? y - 25 : 0, textWidth + 10, 25);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(
          `${prediction.class} (${Math.round(prediction.score * 100)}%)`,
          x + 5, y > 20 ? y - 8 : 17
        );

        newObjectCounts[prediction.class] = (newObjectCounts[prediction.class] || 0) + 1;
      });

      // Run custom model detection if available
      if (customModel) {
        try {
          const customDetections = await detectWithCustomModel(canvas);
          customDetections.forEach((detection, index) => {
            const [x, y, width, height] = detection.bbox;
            ctx.strokeStyle = '#FFFF00';
            ctx.fillStyle = '#FFFF00';
            ctx.strokeRect(x, y, width, height);
            
            // Draw label with background
            const textWidth = ctx.measureText(`${detection.class} (${Math.round(detection.score * 100)}%)`).width;
            ctx.fillRect(x, y > 25 ? y - 25 : 0, textWidth + 10, 25);
            ctx.fillStyle = '#000000';
            ctx.fillText(
              `${detection.class} (${Math.round(detection.score * 100)}%)`,
              x + 5, y > 20 ? y - 8 : 17
            );

            newObjectCounts[detection.class] = (newObjectCounts[detection.class] || 0) + 1;
            predictions.push(detection);
          });
        } catch (err) {
          console.error("Error running custom model:", err);
        }
      }

      setObjectCounts(newObjectCounts);
      setDetections(predictions);
      
      // Track detection history for analysis
      if (predictions.length > 0) {
        setDetectionHistory(prev => {
          const newHistory = [
            ...prev, 
            {
              timestamp: new Date().toISOString(),
              detections: predictions.map(p => ({ 
                class: p.class, 
                confidence: p.score 
              }))
            }
          ];
          return newHistory.slice(-100); // Keep last 100 records
        });
      }

      // Calculate FPS
      if (mode === 'detect') {
        const now = Date.now();
        const elapsed = now - lastDetectionTime.current;
        frameCount.current++;
        if (elapsed > 1000) {
          setFps(Math.round((frameCount.current * 1000) / elapsed));
          frameCount.current = 0;
          lastDetectionTime.current = now;
        }

        // Continue the detection loop
        animationRef.current = requestAnimationFrame(() => detectObjects());
      }
    } catch (err) {
      console.error("Error during detection:", err);
      
      // Try to recover from transient errors
      if (mode === 'detect') {
        setTimeout(() => {
          console.log("Attempting to recover from detection error...");
          animationRef.current = requestAnimationFrame(() => detectObjects());
        }, 1000);
      }
    }
  }, [model, mode, confidenceThreshold, customModel]);

  // Detect with custom trained model
  const detectWithCustomModel = async (imageData) => {
    if (!customModel) return [];

    try {
      // Convert image to tensor
      const tensor = tf.browser.fromPixels(imageData);
      const normalized = tensor.div(255.0).expandDims(0);
      
      // Run prediction
      const predictions = await customModel.predict(normalized);
      
      // Process predictions into detection format
      const classes = Object.keys(trainingImages);
      const scores = predictions.dataSync();
      const detections = [];
      
      for (let i = 0; i < classes.length; i++) {
        if (scores[i] > confidenceThreshold) {
          detections.push({
            class: classes[i],
            score: scores[i],
            bbox: [0, 0, imageData.width, imageData.height] // Default to full image
          });
        }
      }
      
      // Cleanup tensors
      tensor.dispose();
      normalized.dispose();
      predictions.dispose();
      
      return detections;
    } catch (err) {
      console.error("Error in custom detection:", err);
      return [];
    }
  };

  // Get consistent color for object classes
  const getColorForClass = (className) => {
    const colorMap = {
      person: '#FF0000',
      car: '#00FF00',
      dog: '#0000FF',
      cat: '#FFFF00',
      bicycle: '#FF00FF',
      chair: '#00FFFF',
      laptop: '#FFA500',
      'cell phone': '#800080',
      book: '#008000',
      tv: '#FFC0CB',
    };

    if (colorMap[className]) {
      return colorMap[className];
    }

    // Generate color for unknown classes
    let hash = 0;
    for (let i = 0; i < className.length; i++) {
      hash = className.charCodeAt(i) + ((hash << 5) - hash);
    }
    const color = '#' + ('000000' + (hash & 0xFFFFFF).toString(16)).slice(-6);
    return color;
  };

  // Handler for changing mode (detect/capture/train)
  const handleModeChange = (newMode) => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    
    setMode(newMode);
    
    if (newMode === 'detect' && model && isVideoPlaying) {
      // Reset stats before starting detection
      setDetections([]);
      setObjectCounts({});
      
      // Start detection with a slight delay to ensure UI has updated
      setTimeout(() => {
        detectObjects();
      }, 100);
    } else {
      setDetections([]);
      setObjectCounts({});
    }
  };

  // Handle image capture for training
  const handleCaptureImage = () => {
    if (videoRef.current && captureCanvasRef.current && currentTrainingClass) {
      const video = videoRef.current;
      const canvas = captureCanvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      const imageDataUrl = canvas.toDataURL('image/jpeg');
      setCapturedImage(imageDataUrl);
      
      const img = new Image();
      img.onload = () => {
        // Add to training images
        setTrainingImages(prev => ({
          ...prev,
          [currentTrainingClass]: [...(prev[currentTrainingClass] || []), img]
        }));
        
        setTrainingLogs(prev => [
          ...prev, 
          `Captured image for class "${currentTrainingClass}". Total: ${
            (trainingImages[currentTrainingClass]?.length || 0) + 1
          }`
        ]);
      };
      img.src = imageDataUrl;
    } else {
      alert("Please select a class to train first");
    }
  };

  // Add new custom class
  const handleAddClass = () => {
    if (newClassName && !customClasses.includes(newClassName)) {
      setCustomClasses([...customClasses, newClassName]);
      setTrainingLogs(prev => [...prev, `Added new class: ${newClassName}`]);
      setNewClassName('');
    }
  };

  // Handle image upload from file for training
  const handleImageUpload = (e, className) => {
    const files = e.target.files;
    if (files.length === 0) return;
    
    setTrainingLogs(prev => [...prev, `Processing ${files.length} image(s) for class "${className}"...`]);
    
    Array.from(files).forEach(file => {
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
    });
    
    setTrainingLogs(prev => [...prev, `Images for class "${className}" processed and added to training set`]);
  };

  // Set class for capturing training images
  const handleSetTrainingClass = (className) => {
    setCurrentTrainingClass(className);
    setTrainingLogs(prev => [...prev, `Selected "${className}" for training capture`]);
  };

  // Train custom model on captured images
  const trainCustomModel = async () => {
    try {
      setIsTraining(true);
      setTrainingProgress(0);
      setTrainingLogs(prev => [...prev, "Starting model training..."]);
      
      // Check if we have enough training data
      const classes = Object.keys(trainingImages);
      if (classes.length < 2) {
        throw new Error("Need at least 2 classes for training");
      }
      
      for (const className of classes) {
        if (!trainingImages[className] || trainingImages[className].length < 5) {
          throw new Error(`Need at least 5 images for class "${className}"`);
        }
      }
      
      // Create model architecture
      const model = tf.sequential();
      model.add(tf.layers.conv2d({
        inputShape: [224, 224, 3],
        filters: 16,
        kernelSize: 3,
        activation: 'relu'
      }));
      model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
      model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
      }));
      model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
      model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu'
      }));
      model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
      model.add(tf.layers.flatten());
      model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
      model.add(tf.layers.dense({ units: classes.length, activation: 'softmax' }));
      
      // Compile model
      model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
      
      // Prepare training data
      setTrainingLogs(prev => [...prev, "Processing training data..."]);
      const xs = [];
      const ys = [];
      
      for (let i = 0; i < classes.length; i++) {
        const className = classes[i];
        const images = trainingImages[className] || [];
        
        for (const img of images) {
          try {
            // Process image
            const tensor = tf.browser.fromPixels(img);
            const resized = tf.image.resizeBilinear(tensor, [224, 224]);
            const normalized = resized.div(255.0);
            xs.push(normalized);
            
            // Create one-hot label
            const label = new Array(classes.length).fill(0);
            label[i] = 1;
            ys.push(label);
            
            tensor.dispose();
            resized.dispose();
          } catch (err) {
            console.error("Error processing image:", err);
          }
        }
      }
      
      if (xs.length === 0) {
        throw new Error("No valid training images processed");
      }
      
      setTrainingLogs(prev => [...prev, `Processed ${xs.length} training images across ${classes.length} classes`]);
      
      // Stack all examples into tensors
      const xTensor = tf.stack(xs);
      const yTensor = tf.tensor2d(ys);
      
      // Train model
      setTrainingLogs(prev => [...prev, "Beginning model training (this may take a while)..."]);
      await model.fit(xTensor, yTensor, {
        epochs: 10,
        batchSize: 16,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            const progress = Math.round(((epoch + 1) / 10) * 100);
            setTrainingProgress(progress);
            setTrainingLogs(prev => [
              ...prev, 
              `Epoch ${epoch+1}/10 - loss: ${logs.loss.toFixed(4)} - accuracy: ${logs.acc.toFixed(4)}`
            ]);
          }
        }
      });
      
      // Cleanup tensors
      xTensor.dispose();
      yTensor.dispose();
      
      // Save model
      setCustomModel(model);
      setTrainingLogs(prev => [...prev, "Model training complete! Custom model is now active."]);
      
      // Optional - save to browser storage
      try {
        await model.save('indexeddb://custom-object-detection-model');
        setTrainingLogs(prev => [...prev, "Model saved to browser storage"]);
      } catch (saveErr) {
        console.error("Error saving model:", saveErr);
        setTrainingLogs(prev => [...prev, `Couldn't save model to browser storage: ${saveErr.message}`]);
      }
      
    } catch (err) {
      console.error("Training error:", err);
      setTrainingLogs(prev => [...prev, `ERROR: ${err.message}`]);
    } finally {
      setIsTraining(false);
    }
  };

  // Load previously saved model
  const loadCustomModel = async () => {
    try {
      setTrainingLogs(prev => [...prev, "Loading saved model from browser storage..."]);
      const model = await tf.loadLayersModel('indexeddb://custom-object-detection-model');
      setCustomModel(model);
      setTrainingLogs(prev => [...prev, "Custom model loaded successfully!"]);
    } catch (err) {
      console.error("Error loading saved model:", err);
      setTrainingLogs(prev => [...prev, `Error loading model: ${err.message}`]);
    }
  };

  // Clear all training data
  const clearTrainingData = () => {
    if (window.confirm("Are you sure you want to clear all training data? This cannot be undone.")) {
      setTrainingImages({});
      setTrainingLogs(prev => [...prev, "All training data cleared"]);
    }
  };

  // Effect to run detection in detection mode when model or video status changes
  useEffect(() => {
    if (mode === 'detect' && model && isVideoPlaying && !modelLoading) {
      console.log("Starting detection due to model/video ready");
      detectObjects();
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [mode, model, detectObjects, isVideoPlaying, modelLoading]);

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="App">
      <h1>AI Object Detection & Custom Training</h1>
      
      {/* Mode Selector */}
      <div className="mode-selector">
        <button 
          className={mode === 'detect' ? 'active' : ''} 
          onClick={() => handleModeChange('detect')}
        >
          Object Detection
        </button>
        <button 
          className={mode === 'capture' ? 'active' : ''} 
          onClick={() => handleModeChange('capture')}
        >
          Training Capture
        </button>
        <button 
          className={mode === 'train' ? 'active' : ''} 
          onClick={() => handleModeChange('train')}
        >
          Model Training
        </button>
      </div>
      
      {/* Video and Canvas Display */}
      <div className="video-container">
        {modelLoading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Loading AI model...</p>
          </div>
        )}
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
        <canvas
          ref={captureCanvasRef}
          style={{ display: 'none' }}
        />
      </div>
      
      {/* DETECTION MODE UI */}
      {mode === 'detect' && (
        <div className="detection-panel">
          <div className="detection-stats">
            <h2>Detection Stats</h2>
            <p>FPS: {fps}</p>
            
            <button 
              onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
              className="options-button"
            >
              {showAdvancedOptions ? 'Hide' : 'Show'} Advanced Options
            </button>
            
            {showAdvancedOptions && (
              <div className="advanced-options">
                <label>
                  Confidence Threshold: {confidenceThreshold}
                  <input 
                    type="range" 
                    min="0.1" 
                    max="0.9" 
                    step="0.05" 
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  />
                </label>
              </div>
            )}
            
            <h3>Object Counts:</h3>
            <ul className="object-list">
              {Object.entries(objectCounts).map(([objectClass, count]) => (
                <li 
                  key={objectClass} 
                  onClick={() => setSelectedObject(objectClass)} 
                  className={selectedObject === objectClass ? 'selected' : ''}
                >
                  {objectClass}: {count}
                </li>
              ))}
            </ul>
            
            {detections.length === 0 && !modelLoading && (
              <div className="no-detections">
                <p>No objects detected. Try:</p>
                <ul>
                  <li>Showing objects more clearly to the camera</li>
                  <li>Adjusting the confidence threshold</li>
                  <li>Ensuring good lighting conditions</li>
                </ul>
              </div>
            )}
          </div>
          
          {selectedObject && (
            <div className="object-details">
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
        </div>
      )}
      
      {/* CAPTURE MODE UI */}
      {mode === 'capture' && (
        <div className="capture-panel">
          <h2>Training Data Capture</h2>
          <p>Capture images for custom object detection training</p>
          
          <div className="class-selector">
            <h3>Select Class to Train:</h3>
            <div className="class-buttons">
              {customClasses.map(className => (
                <button 
                  key={className}
                  onClick={() => handleSetTrainingClass(className)}
                  className={currentTrainingClass === className ? 'active' : ''}
                >
                  {className} ({trainingImages[className]?.length || 0})
                </button>
              ))}
            </div>
            
            <div className="add-class">
              <input
                type="text"
                value={newClassName}
                onChange={(e) => setNewClassName(e.target.value)}
                placeholder="Enter new class name"
              />
              <button onClick={handleAddClass}>Add Class</button>
            </div>
          </div>
          
          {currentTrainingClass && (
            <div className="capture-controls">
              <h3>Capturing for: {currentTrainingClass}</h3>
              <p>Current images: {trainingImages[currentTrainingClass]?.length || 0}</p>
              <button 
                className="capture-button"
                onClick={handleCaptureImage}
              >
                Capture Image
              </button>
              
              <div className="file-upload">
                <h4>Or upload images:</h4>
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={(e) => handleImageUpload(e, currentTrainingClass)}
                />
              </div>
              
              {capturedImage && (
                <div className="last-capture">
                  <h4>Last Captured:</h4>
                  <img src={capturedImage} alt="Captured" />
                </div>
              )}
            </div>
          )}
        </div>
      )}
      
      {/* TRAINING MODE UI */}
      {mode === 'train' && (
        <div className="training-panel">
          <h2>Model Training</h2>
          
          <div className="training-status">
            <h3>Training Data Summary:</h3>
            <ul>
              {customClasses.map(className => (
                <li key={className}>
                  {className}: {trainingImages[className]?.length || 0} images
                </li>
              ))}
            </ul>
            
            {isTraining ? (
              <div className="training-progress">
                <h3>Training in Progress: {trainingProgress}%</h3>
                <progress value={trainingProgress} max="100" />
              </div>
            ) : (
              <div className="training-actions">
                <button 
                  onClick={trainCustomModel}
                  disabled={customClasses.length < 2}
                  className="train-button"
                >
                  Train Custom Model
                </button>
                <button onClick={loadCustomModel}>
                  Load Saved Model
                </button>
                <button onClick={clearTrainingData} className="danger-button">
                  Clear All Training Data
                </button>
              </div>
            )}
          </div>
          
          <div className="training-logs">
            <h3>Training Logs:</h3>
            <div className="logs-container">
              {trainingLogs.map((log, index) => (
                <div key={index} className="log-entry">
                  {log}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;