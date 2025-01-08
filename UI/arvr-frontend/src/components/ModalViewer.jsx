import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";

const ModelViewer = ({ modelPath }) => {
  const viewerRef = useRef();

  useEffect(() => {
    if (!modelPath) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    viewerRef.current.appendChild(renderer.domElement);

    const loader = new GLTFLoader();
    loader.load(
      modelPath,
      (gltf) => {
        scene.add(gltf.scene);
        animate();
      },
      undefined,
      (error) => {
        console.error("Error loading model:", error);
      }
    );

    const animate = () => {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };

    return () => {
      viewerRef.current.removeChild(renderer.domElement);
    };
  }, [modelPath]);

  return <div ref={viewerRef} />;
};

export default ModelViewer;
