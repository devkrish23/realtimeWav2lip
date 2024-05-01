document.getElementById('fileInput').addEventListener('change', function() {
  var selectedFile = this.files[0];
  document.getElementById('selectedFileName').innerText = selectedFile ? selectedFile.name : 'No file selected';
  //document.getElementById('fileText').innerText = selectedFile ? 'File Selected: ' + selectedFile.name : '';
  
  // Send the selected file to the server using AJAX
  var formData = new FormData();
  formData.append('image', selectedFile);
  
  fetch('/upload', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    if (response.ok) {
      console.log('File uploaded successfully');
    } else {
      console.error('Error uploading file');
    }
  })
  .catch(error => {
    console.error('Error:', error);
  });
});

