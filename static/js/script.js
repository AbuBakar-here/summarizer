const fileInput = document.querySelector('input[type="file"]');
const fileNameDisplay = document.querySelector('#file-name');

fileInput.addEventListener('change', (event) => {
    const fileName = event.target.files[0].name;
    fileNameDisplay.textContent = fileName;
    });
