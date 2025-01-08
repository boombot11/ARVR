import React, { useState } from "react";

const App = () => {
  const [files, setFiles] = useState(null);
  const [responseMessage, setResponseMessage] = useState(null);

  const uploadImages = async () => {
    if (!files || files.length === 0) {
      alert("Please select files");
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("file", files[i]); // The key "files" should match FastAPI's expected parameter name
    }

    try {
      const response = await fetch("http://localhost:8000/upload/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      
      if (response.ok) {
        setResponseMessage(data.message);
      } else {
        setResponseMessage(data.detail || "Error uploading files.");
      }
    } catch (error) {
      setResponseMessage("Error during file upload: " + error.message);
    }
  };

  return (
    <div>
      <h1>Upload Image to Generate 3D Model</h1>
      <input
        type="file"
        onChange={(e) => setFiles(e.target.files)}
        accept="image/jpeg, image/png"
        multiple
      />
      <button onClick={uploadImages}>Upload Image</button>

      {responseMessage && <div>{responseMessage}</div>}
    </div>
  );
};

export default App;
