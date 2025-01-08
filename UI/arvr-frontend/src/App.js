import React, { useState } from 'react';

const App = () => {
    const [files, setFiles] = useState(null);
    const [modelUrl, setModelUrl] = useState(null);

    // Handle image upload and send to backend
    const uploadImages = async () => {
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append("file", files[i]);
        }

        const response = await fetch("http://localhost:8000/upload/", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        if (data.model_file) {
            setModelUrl(data.model_file);  // Access model_file instead of model_path
        } else {
            alert('Error generating model');
        }
    };

    return (
        <div>
            <h1>Upload Image to Generate 3D Model</h1>
            <input type="file" onChange={(e) => setFiles(e.target.files)} />
            <button onClick={uploadImages}>Upload Image</button>
            {modelUrl && (
                <div>
                    <h3>Generated 3D Model</h3>
                    <iframe
                        title="3D Model Viewer"
                        src={`http://localhost:8000${modelUrl}`}  // Assuming the model file is accessible via this URL
                        style={{ width: "600px", height: "400px" }}
                    ></iframe>
                </div>
            )}
        </div>
    );
};

export default App;
