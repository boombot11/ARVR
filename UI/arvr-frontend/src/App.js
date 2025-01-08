import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { fromArrayBuffer } from 'numpy-parser'; // Importing fromArrayBuffer for parsing the .npy file
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';// Import OrbitControls

const App = () => {
    const [files, setFiles] = useState(null);
    const [modelData, setModelData] = useState(null);
    const canvasRef = useRef(null);  // Ref for Three.js canvas

    // Handle image upload and send to backend
    const uploadImages = async () => {
        if (!files || files.length === 0) {
            alert('Please select at least one image');
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('file', files[i]);
        }

        try {
            const response = await fetch("http://localhost:8000/upload/", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Failed to upload images');
            }

            const data = await response.json();
            console.log(data);
        
            if (data.model_file) {
                // Fetch the .npy file and pass it to the 3D model rendering
                const modelResponse = await fetch(`http://localhost:8000${data.model_file}`);
                const modelData = await modelResponse.arrayBuffer();
                setModelData(modelData);  // Save model data for rendering
            } else {
                alert('Error generating model');
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    };

    useEffect(() => {
        if (!modelData) return;
    
        // Parse the .npy data using fromArrayBuffer synchronously
        const parsedData = fromArrayBuffer(modelData);  // Directly get parsed data
        console.log('Parsed Data:', parsedData); // Log the parsed data
    
        if (parsedData && parsedData.data) {
            // Assuming the shape is [1, 32, 32, 32] (1 sample, 32x32x32 grid)
            const voxelGrid = parsedData.data;  // This is a flat Float32Array
            const size = 32;  // Based on the shape, we know it's 32x32x32 grid
            const threshold = 0.1;  // Threshold for determining if a voxel is 'on'
    
            // Set up Three.js scene, camera, and renderer
            const canvas = canvasRef.current;
            const renderer = new THREE.WebGLRenderer({ canvas });
            renderer.setSize(window.innerWidth, window.innerHeight);
    
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.z = 5;  // Adjust camera distance for your model
    
            // Create material for voxel cubes
            const voxelMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
    
            // Loop through the voxel grid and create cubes for non-zero values
            for (let x = 0; x < size; x++) {
                for (let y = 0; y < size; y++) {
                    for (let z = 0; z < size; z++) {
                        // Access the voxel value using a flattened index
                        const index = x * size * size + y * size + z;
                        const value = voxelGrid[index];  // Get the voxel value at the flattened index
    
                        // If the value is above a threshold, display the voxel
                        if (value > threshold) {
                            // Create a small cube for each non-zero voxel
                            const voxelGeometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);  // Adjust size as needed
                            const voxelMesh = new THREE.Mesh(voxelGeometry, voxelMaterial);
    
                            // Position the cube at the voxel grid location
                            voxelMesh.position.set(x * 0.12, y * 0.12, z * 0.12);  // Adjust spacing as needed
                            scene.add(voxelMesh);
                        }
                    }
                }
            }
    
            // Add OrbitControls
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;  // Smooth control
            controls.dampingFactor = 0.25;  // Damping factor for smooth transitions
            controls.screenSpacePanning = false; // Disables panning along the screen plane
            controls.maxPolarAngle = Math.PI / 2; // Limit vertical rotation to 90 degrees
    
            // Animation loop to render the scene and update the camera
            const animate = () => {
                requestAnimationFrame(animate);
                controls.update();  // Update the controls every frame
                renderer.render(scene, camera);
            };
    
            animate();
    
            // Clean up on component unmount
            return () => {
                renderer.dispose();
                scene.dispose();
                controls.dispose();  // Dispose of the controls when unmounting
            };
        } else {
            console.error('Parsed data or parsedData.data is undefined');
        }
    }, [modelData]);
    
    return (
        <div>
            <h1>Upload Images to Generate 3D Model</h1>

            <input 
                type="file" 
                accept="image/*" 
                onChange={(e) => setFiles(e.target.files)} 
                multiple 
            />
            
            <button onClick={uploadImages}>Upload Images</button>

            {/* Canvas for Three.js rendering */}
            <canvas ref={canvasRef}></canvas>
        </div>
    );
};

export default App;
