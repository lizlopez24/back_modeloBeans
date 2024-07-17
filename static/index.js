document.getElementById('uploadButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file!');
        return;
    }
 
    const formData = new FormData();
    formData.append('file', file);
 
    const response = await fetch('/modelo', {
        method: 'POST',
        body: formData
    });
 
    const result = await response.json();
    document.getElementById('result').innerText = JSON.stringify(result);
});