import React, { useState } from 'react';

const App = () => {
    const [files, setFiles] = useState(null);
    const [modelUrl, setModelUrl] = useState(null);

    // Handle image upload and send to backend
    const uploadImages = async () => {
        if (!files || files.length === 0) {
            alert("Please select images to upload.");
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append("file", files[i]);
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
            if (data.model_file) {
                setModelUrl(data.model_file);  // Access model_file returned by backend
            } else {
                alert('Error generating model');
            }
        } catch (error) {
            console.error(error);
            alert('Error occurred while uploading images');
        }
    };

    return (
        <div>
            <h1>Upload Images to Generate 3D Model</h1>
            <input 
                type="file" 
                onChange={(e) => setFiles(e.target.files)} 
                multiple  // Allow multiple file selection
            />
            <button onClick={uploadImages}>Upload Images</button>
            {modelUrl && (
                <div>
                    <h3>Generated 3D Model</h3>
                    <iframe
                        title="3D Model Viewer"
                        src={`http://localhost:8000${modelUrl}`}  // Assuming model file is accessible via this URL
                        style={{ width: "600px", height: "400px" }}
                    ></iframe>
                </div>
            )}
        </div>
    );
};

export default App;
