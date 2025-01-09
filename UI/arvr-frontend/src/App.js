import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; // Import OrbitControls
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'; // Import GLTFLoader

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
            console.log('Response from server:', data);

            if (data.model_file) {
                // Fetch the .glb file and pass it to the 3D model rendering
                const modelResponse = await fetch(`http://localhost:8000${data.model_file}`);
                const modelData = await modelResponse.blob();  // Get as blob for .glb
                setModelData(modelData);  // Save model data for rendering
            } else {
                alert('Error generating model');
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
            console.error('Error during image upload:', error);
        }
    };

    useEffect(() => {
        if (!modelData) return;

        // Set up Three.js scene, camera, and renderer
        const canvas = canvasRef.current;
        const renderer = new THREE.WebGLRenderer({ canvas });
        renderer.setSize(window.innerWidth, window.innerHeight);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;  // Adjust camera distance for your model

        // Add basic ambient and directional lights to ensure model is visible
        const ambientLight = new THREE.AmbientLight(0x404040, 2);  // Ambient light
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);  // Directional light
        directionalLight.position.set(1, 1, 1).normalize();
        scene.add(directionalLight);

        // Load the glTF model using GLTFLoader
        const loader = new GLTFLoader();
        const url = URL.createObjectURL(modelData); // Convert blob to URL
        loader.load(url, (gltf) => {
            scene.add(gltf.scene);  // Add the loaded model to the scene
            console.log('Model loaded:', gltf.scene);

            // Debugging: Check if the model contains objects
            if (gltf.scene.children.length === 0) {
                console.error('The loaded model contains no children');
            } else {
                console.log('Loaded model has children:', gltf.scene.children);
            }

            // Optional: Scale the model if it's too large or small
            gltf.scene.scale.set(1, 1, 1);
        }, undefined, (error) => {
            console.error('Error loading model:', error);
        });

        // Add OrbitControls for easy navigation
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
